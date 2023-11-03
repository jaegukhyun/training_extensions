"""OTX adapters.torch.mmengine.mmpretrain.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from otx.v2.api.core.dataset import BaseDataset
from otx.v2.api.utils import set_tuple_constructor
from otx.v2.api.utils.decorators import add_subset_dataloader

SUBSET_LIST = ["train", "val", "test"]


@add_subset_dataloader(SUBSET_LIST)
class BaseTorchDataset(BaseDataset):
    """A class representing a dataset for pretraining a model."""

    def _initialize(self) -> None:
        self.set_datumaro_adapters()  # Set self.dataset_entity & self.label_schema
        self.initialize = True

    def _build_dataset(
        self,
        subset: str,
        pipeline: list | None = None,
        config: dict | None = None,
    ) -> TorchDataset | None:
        """Builds a TorchDataset object for the given subset using the specified pipeline and configuration.

        Args:
            subset (str): The subset to build the dataset for.
            pipeline (list | None, optional): The pipeline to use for the dataset.
                Defaults to None.
            config (dict | None, optional): The configuration to use for the dataset.
                Defaults to None.

        Returns:
            TorchDataset | None: The built TorchDataset object, or None if the dataset is empty.

        Examples:
        >>> dataset._build_dataset(subset="train")
        torch.utils.data.Dataset()
        >>> dataset._build_dataset(
            subset="train",
            pipeline=[dict(type="Resize", scale=[224, 224])],
        )
        Dataset with Resize pipeline
        """
        raise NotImplementedError


    def _build_dataloader(
        self,
        dataset: TorchDataset | None,
        batch_size: int | None = 2,
        num_workers: int | None = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler: Sampler | (Iterable | dict) | None = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> TorchDataLoader | None:
        """Builds a PyTorch DataLoader for the given dataset.

        Args:
            dataset (Optional[TorchDataset]): The dataset to load.
            batch_size (int): The batch size to use.
            num_workers (int): The number of worker processes to use for data loading.
            shuffle (bool): Whether to shuffle the data.
            pin_memory (bool): Whether to pin memory for faster GPU transfer.
            drop_last (bool): Whether to drop the last incomplete batch.
            sampler (Optional[Union[Sampler, Iterable, Dict]]): The sampler to use for data loading.
            persistent_workers (bool): Whether to keep the worker processes alive between iterations.
            **kwargs: This can take more input provided by ``torch.utils.data.DataLoader`` via kwargs.

        Returns:
            Optional[TorchDataLoader]: The DataLoader for the given dataset.

        Example:
        >>> torch_dataset
        torch.utils.data.Dataset()
        >>> dataset._build_datloader(dataset=torch_dataset)
        torch.utils.data.DataLoader()
        """
        if dataset is None:
            return None

        # Sampler
        if isinstance(sampler, dict):
            pass
        if sampler is not None:
            shuffle = False

        dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            **kwargs,
        )
        sampler_config: dict | list | None = None
        if isinstance(sampler, dict):
            sampler_config = sampler
        elif isinstance(sampler, Sampler):
            sampler_class_name = getattr(sampler.__class__, '__qualname__', None)
            sampler_config = {"type": sampler_class_name} if sampler_class_name else {"type": str(sampler.__class__)}
        elif isinstance(sampler, Iterable):
            sampler_config = []
            for s in sampler:
                sampler_class_name = getattr(s.__class__, '__qualname__', None)
                sampler_config.append(
                    {"type": sampler_class_name} if sampler_class_name else {"type": str(s.__class__)},
                )

        dataset_config  = dataset.configs if hasattr(dataset, "configs") else dataset
        dataloader.configs = {
            "batch_size": batch_size,
            "sampler": sampler_config,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "shuffle": shuffle,
            "persistent_workers": persistent_workers,
            "dataset": dataset_config,
            **kwargs,
        }
        return dataloader

    def subset_dataloader(
        self,
        subset: str,
        pipeline: dict[str, list] | list | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        config: str | dict | None = None,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler: Sampler | (Iterable | dict) | None = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> TorchDataLoader:
        r"""Torch Based Dataset.subset_dataloader.

        This is a method named ``subset_dataloader`` within a class.
        This method is designed to create a PyTorch DataLoader for a specific subset of a dataset.
        The method returns a TorchDataLoader object, which is a PyTorch DataLoader configured
        according to the provided parameters. This DataLoader can then be used to iterate over the specified subset of
        the dataset in the specified batch size, potentially with multiple worker processes and with the data optionally
        reshuffled at every epoch.
        It also provides more versatility through the ``add_subset_dataloader`` decorator function.
        This minimizes duplication of code and provides a more intuitive function.
        Can see a detailed example of this in Example section.

        Args:
            subset (str): Enter an available subset of that dataset.
            pipeline (dict[str, list] | list | None, optional): This can take a dict or a list.
                Case with Dict: In the Semi-SL case,
                    may want to apply a different pipeline between labeled and unlabeled.
                    then we can give a dictionary of the form below.
                    pipeline = {
                        "train": [...],
                        "unlabeled": [...]
                    }
                Dataset Pipeline. Defaults to None.
            batch_size (int | None, optional): How many samples per batch to load. Defaults to None.
            num_workers (int | None, optional): How many subprocesses to use for data loading.
                ``0`` means that the data will be loaded in the main process. Defaults to None.
            config (str | dict | None, optional): Path to configuration file or Config.
                Defaults to None.
            shuffle (bool, optional): Set to ``True`` to have the data reshuffled at every epoch. Defaults to True.
            pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
                into device/CUDA pinned memory before returning them.  If your data elements
                are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
                see the example below. Defaults to False.
            drop_last (bool, optional): value for whether to drop the last data when the batch is not divided up.
                Defaults to False.
            sampler (Sampler | (Iterable | dict) | None, optional): Defines the strategy to draw
                samples from the dataset. Can be any ``Iterable`` with ``__len__``
                implemented. If specified, :attr:`shuffle` must not be specified.. Defaults to None.
            persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
                the worker processes after a dataset has been consumed once. This allows to
                maintain the workers `Dataset` instances alive. Defaults to False.
            **kwargs (Any): Additional arguments to pass to the DataLoader constructor.

        Returns:
            torch.utils.data.DataLoader: Returns a subset of dataLoader.

        Example:
        >>> dataset.subset_dataloader(subset="train")
        Training Dataloader
        >>> dataset.train_dataloader()
        Training Dataloader
        >>> dataset.train_dataloader(batch_size=4)
        Training Dataloader with batch size 4
        >>> dataset.train_dataloader(
            pipeline=[dict(type="Resize", scale=[224, 224])],
        )
        Training Dataloader with Resize pipeline
        >>> dataset.train_dataloader(
            pipeline={
                "train": [...],
                "unlabeled": [...]
            },
        )
        Semi-SL Training DataLoader
        """
        # Config Setting
        _config: dict = {}
        if isinstance(config, str):
            set_tuple_constructor()
            with Path(config).open() as f:
                _config = yaml.safe_load(f)
        elif config is not None:
            _config = config

        subset_pipeline = pipeline
        if isinstance(subset_pipeline, dict):
            subset_pipeline = subset_pipeline[subset]
        subset_dataset = self._build_dataset(subset=subset, pipeline=subset_pipeline, config=_config)
        if batch_size is None:
            batch_size = _config.get("batch_size", 1)
        if num_workers is None:
            num_workers = _config.get("num_workers", 0)

        return self._build_dataloader(
            dataset=subset_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            **kwargs,
        )

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the dataset.

        If the dataset has not been initialized, this method will first initialize it.

        Returns:
            The number of classes in the dataset.
        """
        if not self.initialize:
            self._initialize()
        return len(self.label_schema.get_labels(include_empty=False))