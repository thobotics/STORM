"""Parameter counts of the trained models and output shapes.

    python tests/test_model.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storm import STORM  # noqa: E402

ROCKETMAN = dict(input_size=1, n_nodes=16, window=9, horizon=1, num_blocks=2, step_size=0.3,
                 forcing_style="tanh")
METR_LA = dict(input_size=1, n_nodes=207, window=12, horizon=12, exog_size=10,
               step_size=0.1)
PEMS_BAY = dict(input_size=1, n_nodes=325, window=12, horizon=12, exog_size=9,
                step_size=0.1)
SURF_WAVE = dict(input_size=9, n_nodes=678, window=25, horizon=25, step_size=0.25)

# (config, hidden size, number of parameters)
PAPER_COUNTS = [
    (ROCKETMAN, 128, 514_819),
    (ROCKETMAN, 64, 130_435),
    (ROCKETMAN, 32, 33_475),
    (METR_LA, 128, 327_042),
    (METR_LA, 64, 89_794),
    (METR_LA, 32, 26_466),
    (PEMS_BAY, 128, 341_890),
    (PEMS_BAY, 64, 97_218),
    (PEMS_BAY, 32, 30_178),
    (SURF_WAVE, 128, 387_850),
]


def ring(n: int) -> torch.Tensor:
    i = torch.arange(n)
    return torch.cat([torch.stack([i, (i + 1) % n]), torch.stack([(i + 1) % n, i])], dim=1)


def test_parameter_counts():
    for cfg, hidden, expected in PAPER_COUNTS:
        model = STORM(hidden_size=hidden, **cfg)
        n = sum(p.numel() for p in model.parameters())
        assert n == expected, f"{cfg['n_nodes']} nodes, h={hidden}: {n} != {expected}"


def test_output_shape():
    for cfg in (ROCKETMAN, METR_LA, SURF_WAVE):
        model = STORM(hidden_size=16, **cfg)
        x = torch.randn(2, cfg["window"], cfg["n_nodes"], cfg["input_size"])
        u = torch.randn(2, cfg["window"], cfg["n_nodes"], cfg.get("exog_size", 0))
        y = model(x, ring(cfg["n_nodes"]), u=u if cfg.get("exog_size") else None)
        assert y.shape == (2, cfg["horizon"], cfg["n_nodes"], cfg["input_size"])


if __name__ == "__main__":
    test_parameter_counts()
    test_output_shape()
    print("all tests passed")
