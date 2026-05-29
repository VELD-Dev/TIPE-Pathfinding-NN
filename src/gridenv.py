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
    
    def get_observation(self, agent_idx: int, view_size: int = 11) -> np.ndarray:
        assert view_size % 2 == 1, "view_size doit être impair pour un centrage propre"
        
        n_channels = 5
        obs = np.zeros((n_channels, view_size, view_size), dtype=np.float32)
        
        agent_pos = self.starts[agent_idx]
        agent_goal = self.goals[agent_idx]
        half = view_size // 2
        ar, ac = agent_pos
        
        top, left = ar - half, ac - half
        
        obs[0, :, :] = 1.0
        
        g_top = max(0, top)
        g_left = max(0, left)
        g_bot = min(self.height, top + view_size)
        g_right = min(self.width, left + view_size)
        
        v_top = g_top - top
        v_left = g_left - left
        v_bot = v_top + (g_bot - g_top)
        v_right = v_left + (g_right - g_left)
        
        obs[0, v_top:v_bot, v_left:v_right] = self.grid[g_top:g_bot, g_left:g_right]
        
        # --- Canaux 1 et 2 : autres agents et leurs cibles ---
        for i, (pos, goal) in enumerate(zip(self.starts, self.goals)):
            if i == agent_idx:
                continue
            vr, vc = pos[0] - top, pos[1] - left
            if 0 <= vr < view_size and 0 <= vc < view_size:
                obs[1, vr, vc] = 1.0
            vr, vc = goal[0] - top, goal[1] - left
            if 0 <= vr < view_size and 0 <= vc < view_size:
                obs[2, vr, vc] = 1.0
        
        vr, vc = agent_goal[0] - top, agent_goal[1] - left
        if 0 <= vr < view_size and 0 <= vc < view_size:
            obs[3, vr, vc] = 1.0
        
        dr = agent_goal[0] - ar
        dc = agent_goal[1] - ac
        norm = np.sqrt(dr * dr + dc * dc)
        if norm > 0:
            obs[4, :, :] = dr / norm  # composante verticale
        
        return obs