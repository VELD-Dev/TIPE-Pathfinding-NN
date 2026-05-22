import numpy as np
from collections import deque
from typing import Tuple, List, Optional

class GridEnvironment:
    """
    Environnement 2D de la grille, pour obstacles, etc...
    """

    width: int
    height: int
    obstacle_density: float
    rng:np.random.Generator
    grid: Optional[np.ndarray]
    starts: List[Tuple[int, int]]
    goals: List[Tuple[int, int]]

    def __init__(
            self,
            height: int = 20,
            width: int = 20,
            obstacle_density: float = 0.2,
            seed: Optional[int] = None,
            ):
        self.height = height
        self.width = width
        self.obstacle_density = obstacle_density
        self.rng = np.random.default_rng(seed)

        self.grid = None
        self.starts = []
        self.goals = []

    def _generate_grid(self) -> np.ndarray:
        return (self.rng.random((self.height, self.width)) < self.obstacle_density).astype(np.int8)
    
    def _is_reachable(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        if start == goal:
            return True
        
        visited = np.zeros_like(self.grid, dtype=bool)
        visited[start] = True
        queue = deque([start])

        moves = [(-1,0), (1,0), (0,-1), (0,1)]

        while queue:
            r, c = queue.popleft()
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                
                if(0 <= nr < self.height and 0 <= nc < self.width
                   and not visited[nr, nc] and self.grid[nr, nc] == 0):
                    if (nr, nc) == goal:
                        return True
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        
        return False
    
    def _random_free_cell(self, exclude: set) -> Optional[Tuple[int, int]]:
        free_cells = np.argwhere(self.grid == 0)
        if len(free_cells) == 0:
            return None
        
        candidates = [ (tcell := tuple(cell)) for cell in free_cells if tcell not in exclude ]
        if not candidates:
            return None
        
        idx = self.rng.integers(0, len(candidates))
        return candidates[idx]
    
    def reset(
            self, 
            n_agents: int = 1,
            max_attempts: int = 100
            ) -> bool:
        for attempt in range(max_attempts):
            self.grid = self._generate_grid()
            self.starts = []
            self.goals = []
            occupied = set()

            success = True
            for _ in range(n_agents):
                start = self._random_free_cell(occupied)
                if start is None:
                    success = False
                    break
                occupied.add(start)

                goal = None
                for _ in range(50):
                    candidate = self._random_free_cell(occupied | {start})
                    if candidate is None:
                        break
                    if candidate != start and self._is_reachable(start, candidate):
                        goal = candidate
                        break

                if goal is None:
                    success = False
                    break

                occupied.add(goal)
                self.starts.append(start)
                self.goals.append(goal)
            if success:
                return True
        return False