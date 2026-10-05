"""Train and test STORM.

    python train.py config=rocketman dataset.spatial_order=3 dataset.temporal_distance=5
    python train.py config=surf_wave
    python train.py config=metr_la
    python train.py config=engrad
    python train.py config=pems_bay ckpt=path/to/model.ckpt   # test only

Test metrics are written to <run dir>/test_results.json.
"""
import json
import os

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tsl import logger
from tsl.engines import Predictor
from tsl.experiment import Experiment
from tsl.metrics import torch_metrics

from storm.data.datamodule import SequentialSplitDataModule
from storm.data.rocketman import RocketMan
from storm.data.surf_wave import SurfWave, SurfWaveDataModule, WeightedMaskedMSE
from storm.data.realworld import build_realworld_datamodule
from storm.model import STORM


def build_datamodule(cfg: DictConfig):
    """Returns (datamodule, exogenous size)."""
    d = cfg.dataset
    if d.name == "rocketman":
        dataset = RocketMan(n_nodes=d.n_nodes, n_samples=d.n_samples, window=d.window,
                            horizon=d.horizon, spatial_order=d.spatial_order,
                            temporal_distance=d.temporal_distance,
                            graph_generator=d.graph, seed=d.data_seed)
        dm = SequentialSplitDataModule(dataset, d.val_len, d.test_len,
                                       cfg.batch_size, cfg.workers)
        exog_size = 0
    elif d.name == "surf_wave":
        dataset = SurfWave(d.path, window=d.window, horizon=d.horizon)
        dm = SurfWaveDataModule(dataset, d.val_len, d.test_len, cfg.batch_size, cfg.workers)
        exog_size = 0
    else:
        dm = build_realworld_datamodule(d, cfg.batch_size, cfg.workers)
        exog_size = dm.torch_dataset.input_map.u.shape[-1]
    dm.setup()
    return dm, exog_size


def build_loss(cfg: DictConfig):
    if cfg.loss == "mse":
        return torch_metrics.MaskedMSE()
    if cfg.loss == "mae":
        return torch_metrics.MaskedMAE()
    if cfg.loss == "weighted_mse":
        return WeightedMaskedMSE(grid_weight=cfg.grid_weight, ball_weight=cfg.ball_weight)
    raise ValueError(f"Unknown loss '{cfg.loss}'.")


def run(cfg: DictConfig):
    dm, exog_size = build_datamodule(cfg)

    model_kwargs = dict(input_size=dm.n_channels, output_size=dm.n_channels,
                        n_nodes=dm.n_nodes, window=cfg.dataset.window,
                        horizon=cfg.dataset.horizon, exog_size=exog_size, **cfg.model)

    if cfg.cosine_schedule:
        scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
        scheduler_kwargs = dict(T_max=cfg.epochs, eta_min=1e-6)
    else:
        scheduler_class = scheduler_kwargs = None

    predictor = Predictor(
        model_class=STORM,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs=dict(lr=cfg.lr, weight_decay=cfg.weight_decay),
        loss_fn=build_loss(cfg),
        metrics={"mae": torch_metrics.MaskedMAE(),
                 "mse": torch_metrics.MaskedMSE(),
                 "mape": torch_metrics.MaskedMAPE()},
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=cfg.scale_target,
    )
    n_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    logger.info(f"STORM parameters: {n_params:,}")

    checkpoint = ModelCheckpoint(dirpath=cfg.run.dir, filename="best", save_top_k=1,
                                 monitor=cfg.monitor, mode="min")
    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      default_root_dir=cfg.run.dir,
                      logger=CSVLogger(cfg.run.dir, name="", version=""),
                      accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      devices=1,
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[checkpoint,
                                 EarlyStopping(monitor=cfg.monitor, mode="min",
                                               patience=cfg.patience,
                                               stopping_threshold=cfg.stopping_threshold)])

    if cfg.ckpt is None:
        trainer.fit(predictor, train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())
    ckpt_path = checkpoint.best_model_path if cfg.ckpt is None else cfg.ckpt
    predictor.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    predictor.freeze()

    results = trainer.test(predictor, dataloaders=dm.test_dataloader())[0]
    results["n_params"] = n_params
    with open(os.path.join(cfg.run.dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    Experiment(run_fn=run, config_path="configs").run()
