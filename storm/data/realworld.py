"""METR-LA, PEMS-BAY and EngRAD through tsl (downloaded automatically on first use)."""
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tsl.data import SpatioTemporalDataModule, SpatioTemporalDataset
from tsl.data.preprocessing import MinMaxScaler, StandardScaler
from tsl.datasets import EngRad, MetrLA, PemsBay


def load_dataset(cfg: DictConfig):
    if cfg.name == "metr_la":
        return MetrLA(impute_zeros=True)
    if cfg.name == "pems_bay":
        return PemsBay()
    if cfg.name == "engrad":
        dataset = EngRad(**OmegaConf.to_container(cfg.engrad))
        dataset.mask[:24] = True  # one valid night, so that the min-max range is defined
        return dataset
    raise ValueError(f"Unknown dataset '{cfg.name}'.")


def build_covariates(dataset, cfg: DictConfig) -> np.ndarray:
    """Calendar encodings, optionally the missing-value mask and the dataset's own
    covariates, broadcast to every node and concatenated on the channel axis."""
    u = []
    if cfg.covariates.year:
        u.append(dataset.datetime_encoded("year").values)
    if cfg.covariates.day:
        u.append(dataset.datetime_encoded("day").values)
    if cfg.covariates.weekday:
        u.append(dataset.datetime_onehot("weekday").values)
    if cfg.covariates.mask:
        u.append(dataset.mask.astype(np.float32))
    if "u" in dataset.covariates:
        u.append(dataset.get_frame("u", return_pattern=False))
    ndim = max(c.ndim for c in u)
    return np.concatenate([np.repeat(c[:, None], dataset.n_nodes, 1) if c.ndim < ndim else c
                           for c in u], axis=-1)


def build_realworld_datamodule(cfg: DictConfig, batch_size: int, workers: int = 0
                               ) -> SpatioTemporalDataModule:
    dataset = load_dataset(cfg)
    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        mask=dataset.mask,
        covariates=dict(u=build_covariates(dataset, cfg)),
        connectivity=dataset.get_connectivity(**cfg.connectivity, layout="edge_index"),
        window=cfg.window,
        horizon=cfg.horizon,
        stride=1)

    if cfg.name == "engrad":  # drop windows without any valid target
        keep = torch_dataset[:].mask.float().mean((1, 2, 3)) > 0
        torch_dataset.set_indices(torch_dataset.indices[keep])

    scaler = {"standard": StandardScaler, "minmax": MinMaxScaler}[cfg.scaler]
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=dict(target=scaler(axis=(0, 1))),
        splitter=dataset.get_splitter(**cfg.splitting),
        mask_scaling=True,
        batch_size=batch_size,
        workers=workers)
    dm.setup()

    if cfg.name == "engrad":  # standardise covariates with large values (e.g. temperature)
        u = torch_dataset.u
        axis = list(range(u.ndim - 1))
        mean, std = u[dm.train_slice].mean(axis=axis), u[dm.train_slice].std(axis=axis)
        large = mean > 3
        torch_dataset.u[..., large] = (u[..., large] - mean[..., large]) / std[..., large]
    return dm
