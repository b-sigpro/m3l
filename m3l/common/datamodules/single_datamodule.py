# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from collections.abc import Callable

from torch.utils.data import DataLoader, Dataset, Sampler

import lightning as lt

from aiaccel.torch.datasets import scatter_dataset


class SingleDataModule(lt.LightningDataModule):
    """LightningDataModule for handling single training and validation datasets.

    This module initializes training and validation datasets, wraps them with
    ``scatter_dataset`` if required, and provides corresponding dataloaders.

    Args:
        train_dataset_fn (Callable[..., Dataset[str]]): Function that returns the training dataset.
        val_dataset_fn (Callable[..., Dataset[str]]): Function that returns the validation dataset.
        batch_size (int): Batch size for training.
        val_batch_size (int, optional): Batch size for validation. Defaults to ``batch_size`` if not provided.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 10.
        wrap_scatter_dataset (bool, optional): Whether to wrap datasets with ``scatter_dataset``. Defaults to True.
        prefetch_factor (int | None, optional): Number of batches loaded in advance by each worker. Defaults to None.
        persistent_workers (bool, optional): Whether to keep workers persistent. Defaults to True.
        in_order (bool, optional): Whether to load data in order. Defaults to True.
        pin_memory (bool, optional): Whether to pin memory during data loading. Defaults to False.
        collate_fn (Callable, optional): Custom collate function. Defaults to None.
        train_sampler_fn (Callable[..., Sampler] | None, optional): Function to create a custom sampler
            for the training dataset. If provided, disables shuffling. Defaults to None.

    Methods:
        setup(stage: str | None):
            Prepares datasets for the given stage. Only ``"fit"`` is supported.
        train_dataloader() -> DataLoader:
            Returns the DataLoader for the training dataset.
        val_dataloader() -> DataLoader:
            Returns the DataLoader for the validation dataset.
    """

    def __init__(
        self,
        train_dataset_fn: Callable[..., Dataset[str]],
        val_dataset_fn: Callable[..., Dataset[str]],
        batch_size: int,
        val_batch_size: int | None = None,
        num_workers: int = 10,
        wrap_scatter_dataset: bool = True,
        prefetch_factor: int | None = None,
        persistent_workers: bool = True,
        in_order: bool = True,
        pin_memory: bool = False,
        collate_fn=None,
        train_sampler_fn: Callable[..., Sampler] | None = None,
    ):
        super().__init__()

        self.train_dataset_fn = train_dataset_fn
        self.val_dataset_fn = val_dataset_fn
        self.train_batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size

        self.wrap_scatter_dataset = wrap_scatter_dataset

        self.default_dataloader_kwargs = dict[str, Any](
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            in_order=in_order,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        self.train_sampler_fn = train_sampler_fn

    def setup(self, stage: str | None):
        if stage == "fit":
            if self.wrap_scatter_dataset:
                self.train_dataset = scatter_dataset(self.train_dataset_fn())
                self.val_dataset = scatter_dataset(self.val_dataset_fn())
            else:
                self.train_dataset = self.train_dataset_fn()
                self.val_dataset = self.val_dataset_fn()

            print(f"Dataset size: {len(self.train_dataset)=},  {len(self.val_dataset)=}")
        else:
            raise ValueError("`stage` is not 'fit'.")

    def train_dataloader(self):
        if self.train_sampler_fn is not None:
            sampler = self.train_sampler_fn(self.train_dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            drop_last=True,
            batch_size=self.train_batch_size,
            **self.default_dataloader_kwargs,
            sampler=sampler,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            drop_last=False,
            batch_size=self.val_batch_size,
            **self.default_dataloader_kwargs,
            shuffle=True,
        )
