"""
PINN solver for Einstein-Klein-Gordon (toy version).
"""
import torch, torch.nn as nn

class PINN_EKG(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, hidden))
        layers.append(nn.Tanh())
        for _ in range(depth-1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, r):
        return self.net(r)
