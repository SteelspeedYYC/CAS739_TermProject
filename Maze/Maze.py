# Class to combine grid and feature to become a maze
import numpy as np
from collections import deque


class Maze:
    """
    Maze Object Holding The Grid And Task Definition.

    Encoding:
        0 = Free Cell
        1 = Wall
        2 = Checkpoint

    Start And Goal Are Stored Separately And Are Not Encoded Into The Grid.
    """

    def __init__(self, grid: np.ndarray) -> None:
        # Store Grid As Int8
        self.grid = np.asarray(grid, dtype=np.int8)

        # Size Information
        self.height, self.width = self.grid.shape
        # Max Manhattan Distance Between Two Cells In The Grid
        max_manhattan = (self.height - 1) + (self.width - 1)
        # Use One Third Of It As Default Threshold
        self.MinStartGoalDistance = max_manhattan / 4.0

        # Store Start And Goal
        start, goal = self._sample_start_goal()
        self.start = start
        self.goal = goal

        # Basic Validation
        self._validate_coordinates()


    # Sample Random Start / Goal On Free Cells
    def _sample_start_goal(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Sample Two Distinct Free Cells As Start And Goal.

        Free Cells Are Those With Value == 0.
        A Minimum Manhattan Distance Between Start And Goal
        Is Enforced If MinStartGoalDistance > 0.
        """
        free_positions = np.argwhere(self.grid == 0)
        num_free = free_positions.shape[0]

        if num_free < 2:
            raise ValueError("Not Enough Free Cells To Sample Start And Goal")


        max_attempts = 100

        for _ in range(max_attempts):
            idx = np.random.default_rng().choice(num_free, size=2, replace=False)
            s_r, s_c = free_positions[idx[0]]
            g_r, g_c = free_positions[idx[1]]

            manhattan = abs(int(s_r) - int(g_r)) + abs(int(s_c) - int(g_c))
            if manhattan >= self.MinStartGoalDistance:
                return (int(s_r), int(s_c)), (int(g_r), int(g_c))

        idx = np.random.default_rng().choice(free_positions.shape[0], size=2, replace=False)
        s_r, s_c = free_positions[idx[0]]
        g_r, g_c = free_positions[idx[1]]
        return (int(s_r), int(s_c)), (int(g_r), int(g_c))


    # Basic Validation
    def _validate_coordinates(self) -> None:
        """
        Check That Start And Goal Are Inside Bounds.
        Does Not Enforce Free Cell Here Because You May Want
        To Fix Or Override It In Another Step.
        """
        sr, sc = self.start
        gr, gc = self.goal

        if not (0 <= sr < self.height and 0 <= sc < self.width):
            raise ValueError(f"Start Coordinate {self.start} Is Out Of Bounds")

        if not (0 <= gr < self.height and 0 <= gc < self.width):
            raise ValueError(f"Goal Coordinate {self.goal} Is Out Of Bounds")


    # Simple Structural Features
    def free_ratio(self) -> float:
        """
        Return The Ratio Of Free Cells (Value == 0).
        """
        return float(np.mean(self.grid == 0))

    def wall_ratio(self) -> float:
        """
        Return The Ratio Of Walls (Value == 1).
        """
        return float(np.mean(self.grid == 1))

    def checkpoint_count(self) -> int:
        """
        Return The Number Of Checkpoint Cells (Value == 2).
        """
        return int(np.sum(self.grid == 2))


    # Helper, get all checkpoints for baseline algo to evaluate the maze
    def _get_checkpoints(self) -> list[tuple[int, int]]:
        """
        Return A List Of All Checkpoint Coordinates (Value == 2).
        """
        positions = np.argwhere(self.grid == 2)
        checkpoints: list[tuple[int, int]] = []
        for r, c in positions:
            checkpoints.append((int(r), int(c)))
        return checkpoints

    # Helper, Baseline BFS without consideration about checkpoint
    # Be used in formal baseline BFS
    def _bfs_simple_shortest_path(self) -> int | None:
        """
        Simple BFS From Start To Goal Ignoring Checkpoints.
        """
        sr, sc = self.start
        gr, gc = self.goal

        if self.grid[sr, sc] == 1 or self.grid[gr, gc] == 1:
            return None

        visited = np.zeros((self.height, self.width), dtype=bool)
        q = deque()
        q.append((sr, sc, 0))
        visited[sr, sc] = True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            r, c, dist = q.popleft()
            if (r, c) == (gr, gc):
                return dist

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if not visited[nr, nc] and self.grid[nr, nc] != 1:
                        visited[nr, nc] = True
                        q.append((nr, nc, dist + 1))

        return None
    

    # ------------------------------------------------------------------
    # BFS Shortest Path From Start To Goal Visiting All Checkpoints
    # Modified with AI help
    # ------------------------------------------------------------------

    def _bfs_with_checkpoints(self) -> int | None:
        """
        Run Breadth-First Search From Start To Goal And Require
        That All Checkpoints (Cells With Value == 2) Are Visited
        At Least Once Along The Path.

        State Is (Row, Col, Mask), Where Mask Is A Bitmask Of
        Visited Checkpoints.

        Returns:
            Number Of Steps In The Shortest Valid Path,
            Or None If No Such Path Exists.
        """
        from collections import deque

        sr, sc = self.start
        gr, gc = self.goal

        # Start Or Goal Cannot Be A Wall
        if self.grid[sr, sc] == 1 or self.grid[gr, gc] == 1:
            return None

        # Collect Checkpoints
        checkpoints = self._get_checkpoints()
        k = len(checkpoints)

        # If There Are No Checkpoints, Fallback To Simple BFS
        if k == 0:
            return self._bfs_simple_shortest_path()

        # Map Checkpoint Coordinate -> Bit Index
        cp_index: dict[tuple[int, int], int] = {}
        for i, (r, c) in enumerate(checkpoints):
            cp_index[(r, c)] = i

        full_mask = (1 << k) - 1

        # Initial Mask: If Start Itself Is A Checkpoint, Mark It
        start_mask = 0
        if self.grid[sr, sc] == 2 and (sr, sc) in cp_index:
            start_mask = 1 << cp_index[(sr, sc)]

        # Visited[Row, Col, Mask] = Whether This State Was Seen
        visited = np.zeros(
            (self.height, self.width, 1 << k),
            dtype=bool,
        )

        q = deque()
        q.append((sr, sc, start_mask, 0))  # (Row, Col, Mask, Distance)
        visited[sr, sc, start_mask] = True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            r, c, mask, dist = q.popleft()

            # Goal State: At Goal With All Checkpoints Visited
            if (r, c) == (gr, gc) and mask == full_mask:
                return dist

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    # Cannot Walk Into Walls
                    if self.grid[nr, nc] == 1:
                        continue

                    new_mask = mask
                    # If This Cell Is A Checkpoint, Update Mask
                    if self.grid[nr, nc] == 2:
                        idx = cp_index.get((nr, nc))
                        if idx is not None:
                            new_mask = mask | (1 << idx)

                    if not visited[nr, nc, new_mask]:
                        visited[nr, nc, new_mask] = True
                        q.append((nr, nc, new_mask, dist + 1))

        # No Valid Path That Visits All Checkpoints
        return None
    

    # ------------------------------------------------------------------
    # Structural Openness: Encourage Large Connected Open Areas
    # Modified with AI help
    # ------------------------------------------------------------------

    def _openness_score(self) -> float:
        """
        Compute A Simple Openness Score Based On Local Connectivity.

        Returns:
            A Small Positive Value; Larger Means More Open / Junction-Rich.
        """
        g = self.grid
        h, w = self.height, self.width

        total_walkable = 0
        sum_degree = 0
        high_degree_count = 0  # Cells With Degree >= 3

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(h):
            for c in range(w):
                if g[r, c] == 1:
                    continue  # Wall, Ignore

                total_walkable += 1
                deg = 0

                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if g[nr, nc] != 1:
                            deg += 1

                sum_degree += deg
                if deg >= 3:
                    high_degree_count += 1

        if total_walkable == 0:
            return 0.0

        avg_deg = sum_degree / total_walkable
        junction_ratio = high_degree_count / total_walkable

        return float(0.8 * avg_deg + ((self.height - 1) * (self.width - 1) * self.free_ratio()) * junction_ratio)



    def evaluate_structure(self) -> float:
        """
        Evaluate Maze By Solving It From Start To Goal While Visiting
        All Checkpoints At Least Once.

        If There Is No Such Path, Return A Large Negative Penalty.
        If There Is A Path, Return A Score Based On Path Length.
        """
        path_len = self._bfs_with_checkpoints()

        if path_len is None:
            return -500.0, 1000
        
        cp_count = self.checkpoint_count()
        steps = float(path_len)
        free_ratio = self.free_ratio() 
        space_util_score = 5 * abs(free_ratio - 0.7) # 0.7 is target free cells ratio
        # PN maze always has worse performance, use juction score to buff it a bit with consideration of diversity
        juction_score = self._openness_score()

        # Fitness calculate
        fitness = 2 * steps + 5 * cp_count + 0.5 * juction_score - space_util_score
        return fitness, steps
    

    # THE VERSION THAT BASELINE IGNORE CP AS WELL!!!!
    def evaluate_structure_noCP(self) -> float:
        """
        Evaluate Maze By Solving It From Start To Goal While Visiting
        All Checkpoints At Least Once.

        If There Is No Such Path, Return A Large Negative Penalty.
        If There Is A Path, Return A Score Based On Path Length.
        """
        path_len = self._bfs_simple_shortest_path()

        if path_len is None:
            return -500.0, 1000
        
        cp_count = self.checkpoint_count()
        steps = float(path_len)
        free_ratio = self.free_ratio() 
        space_util_score = 5 * abs(free_ratio - 0.7) # 0.7 is target free cells ratio
        # PN maze always has worse performance, use juction score to buff it a bit with consideration of diversity
        juction_score = self._openness_score()

        # Fitness calculate
        fitness = 2 * steps + 5 * cp_count + 0.5 * juction_score - space_util_score
        return fitness, steps


    # Simple ASCII Visualization, just for Debug, not final presentation
    def to_ascii(self, start_symbol: str = "S", goal_symbol: str = "G") -> str:
        """
        Convert Maze Into An Ascii Representation.

        Mapping:
            0 -> ' '
            1 -> '█'
            2 -> '□'
            Start -> start_symbol
            Goal -> goal_symbol
        """
        mapping = {
            0: " ",
            1: "█",
            2: "□",
        }

        lines: list[str] = []
        for r in range(self.height):
            row_chars: list[str] = []
            for c in range(self.width):
                if (r, c) == self.start:
                    row_chars.append(start_symbol)
                elif (r, c) == self.goal:
                    row_chars.append(goal_symbol)
                else:
                    v = int(self.grid[r, c])
                    row_chars.append(mapping.get(v, "?"))
            lines.append("".join(row_chars))

        return "\n".join(lines)