# STORM: Long-Range Spatio-Temporal Graph Propagation Through Oscillations

Official code of

> **Long-Range Spatio-Temporal Graph Propagation Through Oscillations**<br>
> Alessio Gravina\*, Alessandro Trenta\*, Tai Hoang\*, Andrea Ceni, Gerhard Neumann, Davide Bacciu<br>
> NeurIPS 2026 (\* equal contribution)

**STORM** (Spatio-Temporal Oscillatory dynamics via graph Resistance Model) evolves the node
states $\mathbf{H}(t)$ with a damped, forced wave equation on the graph:

```math
\ddot{\mathbf{H}}(t) = -\widehat{\mathbf{L}}\,\mathbf{H}(t)\,\mathbf{W} \;-\; D\big(\mathbf{H}(t)\big) \odot \dot{\mathbf{H}}(t) \;+\; F\big(\mathbf{X}(t), \mathbf{H}(t)\big)
```

With the velocity $\mathbf{V} = \dot{\mathbf{H}}$ and step size $\epsilon$, one explicit Euler step is

```math
\begin{aligned}
\mathbf{V}_{t+1} &= \mathbf{V}_t - \epsilon \Big( \widehat{\mathbf{L}}\,\mathbf{H}_t \mathbf{W} + D(\mathbf{H}_t) \odot \mathbf{V}_t - F(\mathbf{H}_t, \mathbf{X}_t) \Big), \\
\mathbf{H}_{t+1} &= \mathbf{H}_t + \epsilon\, \mathbf{V}_{t+1},
\end{aligned}
```

where

- $\widehat{\mathbf{L}}$ is the graph Laplacian whose edge weights $a_{uv} = \mathrm{MLP}([\mathbf{h}^u, \mathbf{h}^v])$
  are learned, non-negative edge resistances;
- $D(\mathbf{H}) = \mathrm{ReLU}(\mathbf{H}\mathbf{W}_D)$ is the learned dissipation;
- $F$ is the external forcing that injects the input $\mathbf{X}_t$ (a GCN and an MLP on $[\mathbf{H}_t, \mathbf{X}_t]$).

The initial state $\mathbf{H}_0$ is an encoding of the first input step and $\mathbf{V}_0 = \mathbf{H}_0 \mathbf{W}_V$.
The model reads the input window step by step, keeps integrating for the forecast horizon and
decodes the last states with an MLP. The whole model is in [`storm/model.py`](storm/model.py).

## Repository

```
storm/model.py                  STORM
storm/data/rocketman.py         RocketMan benchmark
storm/data/surf_wave.py         SURF-WAVE dataset and its loss
storm/data/realworld.py         METR-LA, PEMS-BAY and EngRAD (through tsl)
configs/                        one file per experiment, with the paper's hyperparameters
train.py                        trains and tests STORM on any experiment
scripts/generate_surf_wave.py   regenerates SURF-WAVE
scripts/rocketman_matrix.py     success-rate matrices of the RocketMan figures
tests/test_model.py             checks parameter counts and output shapes
```

Training and evaluation are built on [Torch Spatiotemporal (tsl)](https://github.com/TorchSpatiotemporal/tsl)
and [Hydra](https://hydra.cc).

## Installation

Python 3.11 and a CUDA 12.4 GPU. The versions below are the ones used for the paper.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install -r requirements.txt
python tests/test_model.py      # a few seconds on CPU
```

## Data

| Dataset | How to get it |
| --- | --- |
| [RocketMan](https://github.com/marshka/spatiotemporal-oversquashing) | Generated in memory at the start of each run (fixed seed). |
| METR-LA, PEMS-BAY, EngRAD | Downloaded by tsl on first use. |
| SURF-WAVE | Download `surf_wave.h5` (181 MB) from **TODO: link** into `data/`. |

SURF-WAVE can also be regenerated with [NVIDIA Warp](https://github.com/NVIDIA/warp) (GPU):

```bash
python scripts/generate_surf_wave.py --output data/surf_wave.h5
sha256sum data/surf_wave.h5     # TODO: sha256
```

## Reproducing the paper

Every run trains, keeps the checkpoint with the best validation score, tests it and writes
`logs/<dataset>/<time>/test_results.json`. Add `hydra.mode=MULTIRUN seed=0,1,2,3,4` to run five seeds.

| Paper | Command |
| --- | --- |
| Table 1, SURF-WAVE | `python train.py config=surf_wave hydra.mode=MULTIRUN seed=0,1,2,3,4` |
| Table 2, METR-LA | `python train.py config=metr_la hydra.mode=MULTIRUN seed=0,1,2,3,4` |
| Table 2, PEMS-BAY | `python train.py config=pems_bay hydra.mode=MULTIRUN seed=0,1,2,3,4` |
| Table 3, EngRAD | `python train.py config=engrad hydra.mode=MULTIRUN seed=0,1,2,3,4` |
| RocketMan, Figures 2, 5 and 6–9 | see below |
| Figure 10, width on traffic | add `model.hidden_size=64` or `32` to the METR-LA / PEMS-BAY commands |
| Figure 11, width on RocketMan | add `model.hidden_size=64` or `32` to the RocketMan command |
| Table 11, step size | add `model.step_size=0.01,0.02,0.03,0.05,0.1,0.2` |

**RocketMan.** One run is one cell: a graph, a spatial distance *k* and a temporal distance *i*.
The default config is the *L* = 2 model of Appendix F. The full grid is
2 graphs × 81 cells × 10 seeds = 1,620 short runs:

```bash
python train.py config=rocketman hydra.mode=MULTIRUN dataset.graph=ring,lollipop \
    dataset.spatial_order="range(0,9)" dataset.temporal_distance="range(0,9)" seed="range(0,10)"
python scripts/rocketman_matrix.py logs/rocketman
```

**Ablations (Appendix F)** are config switches. Figures 2 and 5 show, for each cell, the best of
these variants.

| Variant | Switch |
| --- | --- |
| *L* = 1 | `model.num_blocks=1` |
| without dissipation | `model.use_dissipation=false` |
| MLP forcing | `model.use_forcing=false` |
| GCN forcing | `model.input_style=none` |
| adaptive resistance | `model.fix_resistance=false` |

**Test a checkpoint** without training:

```bash
python train.py config=metr_la ckpt=logs/metr_la/<time>/best.ckpt
```

Quote paths that contain `=`, e.g. `ckpt="'path/epoch=63-step=19200.ckpt'"`.

## Using STORM in your own code

```python
import torch
from storm import STORM

model = STORM(input_size=1, n_nodes=207, window=12, horizon=12, hidden_size=128, step_size=0.1)
x = torch.randn(8, 12, 207, 1)                 # [batch, window, nodes, channels]
edge_index = torch.randint(0, 207, (2, 1500))  # [2, edges]
y = model(x, edge_index)                       # [batch, horizon, nodes, channels]
```

## Citation

```bibtex
@inproceedings{gravina2026storm,
  title     = {Long-Range Spatio-Temporal Graph Propagation Through Oscillations},
  author    = {Gravina, Alessio and Trenta, Alessandro and Hoang, Tai and Ceni, Andrea and
               Neumann, Gerhard and Bacciu, Davide},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
