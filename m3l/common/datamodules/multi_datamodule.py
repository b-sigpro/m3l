from typing import Any, Callable

from torch.utils.data import DataLoader, Dataset

import lightning as lt
from lightning.pytorch.utilities import CombinedLoader

from aiaccel.torch.datasets import scatter_dataset


class MultiDataModule(lt.LightningDataModule):
    def __init__(
        self,
        train_dataset_fn_dict: dict[str, Callable[..., Dataset[str]]],
        val_dataset_fn_dict: dict[str, Callable[..., Dataset[str]]],
        train_batch_size_list: list[int],
        val_batch_size_list: int,
        num_workers: int = 10,
        wrap_scatter_dataset: bool = True,
    ):
        super().__init__()

        self.train_dataset_fn_dict = train_dataset_fn_dict
        self.val_dataset_fn_dict = val_dataset_fn_dict

        self.train_batch_size_list = train_batch_size_list
        self.val_batch_size_list = val_batch_size_list

        self.default_dataloader_kwargs = dict[str, Any](
            num_workers=num_workers,
            persistent_workers=True,
            shuffle=True,
        )

        self.wrap_scatter_dataset = wrap_scatter_dataset

    def setup(self, stage: str | None):
        if stage == "fit":
            if self.wrap_scatter_dataset:
                self.train_dataset_list = {key: scatter_dataset(fn()) for key, fn in self.train_dataset_fn_dict.items()}
                self.val_dataset_list = {key: scatter_dataset(fn()) for key, fn in self.val_dataset_fn_dict.items()}
            else:
                self.train_dataset = self.train_dataset_fn()
                self.val_dataset = self.val_dataset_fn_dict()

            # print(f"Dataset size: {len(self.train_dataset)=},  {len(self.val_dataset)=}")
        else:
            raise ValueError("`stage` is not 'fit'.")

    def _create_dataloader(self, dataset_list: list[Dataset], batch_size_list: list[int], **kwargs: Any):
        dataloaders = {}
        for (key, ds), bs in zip(dataset_list.items(), batch_size_list):
            dataloaders[key] = DataLoader(ds, bs, **kwargs, **self.default_dataloader_kwargs)

        return CombinedLoader(dataloaders, mode="max_size_cycle")

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset_list, self.train_batch_size_list, drop_last=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset_list, self.val_batch_size_list, drop_last=False)
