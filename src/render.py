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
    cmap_agents = plt.get_cmap('tab10' if n_agents <= 10 else 'tab20')

    if paths is not None:
        for i, path in enumerate(paths):
            if not path:
                continue

            color = cmap_agents(i % cmap_agents.N)
            xs = [c for _, c in path ]
            ys = [r for r, _ in path ]
            ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7, zorder=2)

    for i, (start, goal) in enumerate(zip(env.starts, env.goals)):
        color = cmap_agents(i % cmap_agents.N)
        ax.add_patch(Circle((start[1], start[0]), 0.35, facecolor=color, edgecolor='black', linewidth=1.5, zorder=3))
        ax.add_patch(Circle((goal[1], goal[0]), 0.35, facecolor='none', edgecolor=color, linewidth=2.5, zorder=3))

        if n_agents > 1:
            ax.text(start[1], start[0], str(i), ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=4)
            ax.text(goal[1], goal[0], str(i), ha='center', va='center', fontsize=8, fontweight='bold', color=color, zorder=4)
    
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title)

    return ax

def render_observation(obs: np.ndarray, channel_names: Optional[List[str]] = None) -> None:
    if channel_names is None:
        channel_names = ["Obstacles", "Autres agents", "Cibles autres", "Propre cible", "Direction (dy)", "Direction (dx)"]

    n_channels = obs.shape[0]
    fig, axes = plt.subplots(1, n_channels, figsize=(2.5 * n_channels, 2.8))
    if n_channels == 1:
        axes = [axes]

    for i in range(n_channels):
        vmin, vmax = (-1, -1) if i >= 4 else (0, 1)
        im = axes[i].imshow(obs[i], cmap='RdBu_r' if i >= 4 else "viridis", vmin=vmin, vmax=vmax, interpolation='nearest')
        title = channel_names[i] if i < len(channel_names) else f"Canal {i}"
        axes[i].set_title(title, fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        center = obs.shape[1] // 2
        axes[i].plot(center, center, 'k+', markersize=12, markeredgewidth=2)
    
    plt.tight_layout()
    plt.show()

def animate_paths(env: GridEnvironment, paths: List[List[Tuple[int, int]]], interval: int = 300, save_path: Optional[str] = None) -> FuncAnimation:
    fig, ax = plt.subplots(figsize=(8, 8))
    render_grid(env, paths=paths, ax=ax, title="Trajectoires")
    
    # Longueur maximale pour synchroniser
    max_len = max(len(p) for p in paths)
    # Padding : chaque agent reste à sa dernière position après arrivée
    padded = [p + [p[-1]] * (max_len - len(p)) for p in paths]
    
    cmap_agents = plt.get_cmap('tab10' if len(paths) <= 10 else 'tab20')
    
    # Création des marqueurs mobiles (un par agent)
    markers = []
    for i in range(len(paths)):
        color = cmap_agents(i % cmap_agents.N)
        marker = Circle((0, 0), radius=0.4, facecolor=color, edgecolor='black', linewidth=2, zorder=5)
        ax.add_patch(marker)
        markers.append(marker)
    
    def update(frame):
        for i, marker in enumerate(markers):
            r, c = padded[i][frame]
            marker.center = (c, r)
        ax.set_title(f"Étape {frame}/{max_len - 1}")
        return markers
    
    anim = FuncAnimation(fig, update, frames=max_len, interval=interval, blit=False, repeat=True)
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000 // interval)
        else:
            anim.save(save_path, writer='ffmpeg', fps=1000 // interval)
    
    return anim

if __name__ == "__main__":
    env = GridEnvironment(15, 15, 0.25, 42)

    env.reset(n_agents=1)
    render_grid(env, titre="Carte mono-agent")
    plt.show()

    env.reset(n_agents=4)
    render_grid(env, title="Carte quadri-agent")
    plt.show()

    obs = env.get_observation(agent_idx=0, view_size=11)
    render_observation(obs)

    # à remplacer avec chemins A*
    fake_paths = [ [env.starts[i], env.goals[i]] for i in range(len(env.starts)) ]
