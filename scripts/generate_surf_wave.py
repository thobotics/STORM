"""Generate the SURF-WAVE dataset.

A 2D wave equation is solved with NVIDIA Warp on a 30x30 L-shaped domain (the top-right
quarter is a wall). Three point sources ("balls") start at the bottom-left, at staggered
times, and move right along random sinusoidal paths, displacing the surface as they go.
Sample s uses numpy seed s. 400 simulation steps are stored every 8th step: 50 frames.

    python scripts/generate_surf_wave.py --output data/surf_wave.h5
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import warp as wp

FPS = 60
SUBSTEPS = 16
DT = (1.0 / FPS) / SUBSTEPS
GRID_STEP = 0.1
K_SPEED = 1.0
K_DAMP = 0.0
BALL_RADIUS = 1.5
BALL_MAGNITUDE = 0.5
N_BALLS = 3


@wp.func
def sample(f: wp.array(dtype=float), x: int, y: int, width: int, height: int):
    x = wp.clamp(x, 0, width - 1)
    y = wp.clamp(y, 0, height - 1)
    return f[y * width + x]


@wp.func
def laplacian(f: wp.array(dtype=float), x: int, y: int, width: int, height: int):
    ddx = sample(f, x + 1, y, width, height) - 2.0 * sample(f, x, y, width, height) \
        + sample(f, x - 1, y, width, height)
    ddy = sample(f, x, y + 1, width, height) - 2.0 * sample(f, x, y, width, height) \
        + sample(f, x, y - 1, width, height)
    return ddx + ddy


@wp.kernel
def wave_displace(hcurrent: wp.array(dtype=float), hprevious: wp.array(dtype=float),
                  width: int, height: int, center_x: float, center_y: float,
                  r: float, mag: float, t: float):
    tid = wp.tid()
    dx = float(tid % width) - center_x
    dy = float(tid // width) - center_y
    if dx * dx + dy * dy < r * r:
        h = mag * wp.sin(t)
        hcurrent[tid] = h
        hprevious[tid] = h


@wp.kernel
def wave_solve(hprevious: wp.array(dtype=float), hcurrent: wp.array(dtype=float),
               obstacle: wp.array(dtype=int), width: int, height: int,
               inv_cell: float, k_speed: float, k_damp: float, dt: float):
    tid = wp.tid()
    if obstacle[tid] == 1:
        hprevious[tid] = 0.0
        return
    l = laplacian(hcurrent, tid % width, tid // width, width, height) * inv_cell * inv_cell
    h1 = hcurrent[tid]
    h0 = hprevious[tid]
    hprevious[tid] = 2.0 * h1 - h0 + dt * dt * (k_speed * l - k_damp * (h1 - h0))


def l_shape_mask(size: int) -> np.ndarray:
    """1 = wall (top-right quarter), 0 = water. Indexed [y, x]."""
    mask = np.zeros((size, size), dtype=np.int32)
    mask[size // 2:, size // 2:] = 1
    return mask


class Simulation:

    def __init__(self, size: int, seed: int):
        self.size = size
        np.random.seed(seed)
        self.mask = wp.array(l_shape_mask(size).flatten(), dtype=int)
        self.grid0 = wp.zeros(size * size, dtype=float)
        self.grid1 = wp.zeros(size * size, dtype=float)
        self.balls = self.random_paths()
        self.time = 0.0
        self.step_count = 0

    def random_paths(self) -> list[dict]:
        start_steps = np.random.permutation([0, 50, 100])
        paths = []
        for i in range(N_BALLS):
            start_y = 3 + 3 * i
            max_amplitude = min(start_y - 2, self.size // 2 - start_y - 2)
            paths.append(dict(start_x=3, start_y=start_y, end_x=self.size - 4,
                              amplitude=np.random.uniform(1, max(1.5, max_amplitude)),
                              frequency=np.random.uniform(0.5, 2.0),
                              phase=np.random.uniform(0, 2 * np.pi),
                              start_step=start_steps[i]))
        return paths

    def ball_position(self, i: int, step: int) -> tuple[float, float]:
        """Linear in x, sinusoidal in y; reaches the right side 300 steps after its start."""
        p = self.balls[i]
        if step < p["start_step"]:
            return p["start_x"], p["start_y"]
        progress = min(1.0, (step - p["start_step"]) / 300)
        x = p["start_x"] + progress * (p["end_x"] - p["start_x"])
        y = p["start_y"] + p["amplitude"] * np.sin(2 * np.pi * p["frequency"] * progress
                                                   + p["phase"])
        return np.clip(x, 1, self.size - 2), np.clip(y, 1, self.size - 2)

    def step(self) -> tuple[np.ndarray, list]:
        n = self.size * self.size
        for _ in range(SUBSTEPS):
            for i in range(N_BALLS):
                cx, cy = self.ball_position(i, self.step_count)
                wp.launch(wave_displace, dim=n,
                          inputs=[self.grid0, self.grid1, self.size, self.size, cx, cy,
                                  BALL_RADIUS, BALL_MAGNITUDE, self.time])
            wp.launch(wave_solve, dim=n,
                      inputs=[self.grid1, self.grid0, self.mask, self.size, self.size,
                              1.0 / GRID_STEP, K_SPEED, K_DAMP, DT])
            self.grid0, self.grid1 = self.grid1, self.grid0
            self.time += DT
        wave = self.grid0.numpy().reshape(self.size, self.size)
        balls = [self.ball_position(i, self.step_count) for i in range(N_BALLS)]
        self.step_count += 1
        return wave, balls


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/surf_wave.h5")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--grid_size", type=int, default=30)
    args = parser.parse_args()

    wp.init()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frames = args.steps // args.subsample
    g = args.grid_size
    with h5py.File(args.output, "w") as f:
        wave_ds = f.create_dataset("wave", (args.num_samples, frames, g, g), dtype=np.float32)
        ball_ds = f.create_dataset("balls", (args.num_samples, frames, N_BALLS, 2),
                                   dtype=np.float32)
        f.create_dataset("mask", data=l_shape_mask(g))
        for s in range(args.num_samples):
            sim = Simulation(g, seed=s)
            waves, balls = zip(*(sim.step() for _ in range(args.steps)))
            wave_ds[s] = np.array(waves)[::args.subsample]
            ball_ds[s] = np.array(balls)[::args.subsample]
            if (s + 1) % 50 == 0:
                print(f"{s + 1}/{args.num_samples} samples")
        f.attrs.update(num_samples=args.num_samples, timesteps=frames, grid_size=g,
                       subsample=args.subsample)


if __name__ == "__main__":
    main()
