from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class TrainConfig:
    # grid / physics
    rmax: float = 8.0
    n_colloc: int = 512
    m2L2: float = -2.5
    lamL2: float = 0.5
    L: float = 1.0
    # losses
    w_kg: float = 1.0
    w_tail: float = 1.0
    w_center: float = 1.0
    w_ein: float = 1.0
    w_nec: float = 0.0
    w_dec: float = 0.0
    # optim
    iters_adam: int = 800
    lr: float = 2e-3
    # device
    device: str = "cpu"

    @staticmethod
    def from_yaml(path: str | Path) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text())
        return TrainConfig(**data)
