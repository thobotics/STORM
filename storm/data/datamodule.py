from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, Subset
from tsl.data.loader import StaticGraphLoader


class SequentialSplitDataModule(LightningDataModule):
    """Splits a dataset sequentially into train / val / test and serves tsl loaders.

    Args:
        dataset: the full dataset.
        val_len: validation size, as a fraction (0 < val_len < 1) or a number of samples.
        test_len: test size, as a fraction or a number of samples.
        batch_size: batch size of all loaders.
        workers: number of loader workers.
    """

    def __init__(self, dataset: Dataset, val_len: float = 0.1, test_len: float = 0.2,
                 batch_size: int = 32, workers: int = 0):
        super().__init__()
        self.dataset = dataset
        self._val_len = val_len
        self._test_len = test_len
        self.batch_size = batch_size
        self.workers = workers
        self.train_set = self.val_set = self.test_set = None

    @property
    def n_nodes(self) -> int:
        return self.dataset.n_nodes

    @property
    def n_channels(self) -> int:
        return self.dataset.n_channels

    @property
    def horizon(self) -> int:
        return self.dataset.horizon

    def setup(self, stage: Optional[str] = None):
        n = len(self.dataset)
        n_val = int(n * self._val_len) if 0 < self._val_len < 1 else int(self._val_len)
        n_test = int(n * self._test_len) if 0 < self._test_len < 1 else int(self._test_len)
        n_train = n - n_val - n_test
        self.train_set = Subset(self.dataset, list(range(n_train)))
        self.val_set = Subset(self.dataset, list(range(n_train, n_train + n_val)))
        self.test_set = Subset(self.dataset, list(range(n_train + n_val, n)))

    def _loader(self, subset: Subset, shuffle: bool, drop_last: bool = False):
        return StaticGraphLoader(subset, batch_size=self.batch_size, shuffle=shuffle,
                                 drop_last=drop_last, num_workers=self.workers)

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_set, shuffle=False)
