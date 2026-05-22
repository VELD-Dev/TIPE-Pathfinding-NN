import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple
from gridenv import GridEnvironment

def render_grid(
        env: GridEnvironment,
        paths: Optional[List[List[Tuple[int,int]]]] = None,
        ax: Optional[plt.Axes] = None,
        show_grid_lines: bool = True,
        title: Optional[str] = None
        ) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    cmap = ListedColormap(["white", "#444444"])
    ax.imshow(env.grid, cmap = cmap, vmin = 0, vmax = 1, interpolation="nearest")

    if show_grid_lines:
        ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

    n_agents = len(env.starts)
    cmap_agents = plt.get