import torch
import torch.nn as nn

class Pathfinding(nn.Module):
    """
    Input: vue locale (C, H, W) où C = nb canaux
    - canal 0 = obstacle (1 si mur)
    - canal 1 = autre agent (1 si présent)
    - canal 2 = direction de la cible (gradient ou vecteur)
    Output: logits sur 5 actions (haut, bas, gauche, droite, attendre)
    """

    conv: nn.Sequential
    head: nn.Sequential

    def __init__(self, n_channels:int = 3, view_size: int = 11, n_actions:int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=5),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        flat_size = 64 * view_size * view_size
        self.head = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.head(self.conv(x))
