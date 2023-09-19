"""CheckpointHook with validation results for classification task."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Copyright (c) Open-MMLab. All rights reserved.
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import torch
from mmengine.dist import get_dist_info, master_only
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from torch import distributed as dist
from torch._utils import (
    _flatten_dense_tensors,
    _take_tensors,
    _unflatten_dense_tensors,
)


def allreduce_params(params: List[torch.nn.Parameter], coalesce: bool = True, bucket_size_mb: int = -1) -> None:
    """Allreduce parameters.

    It's from mmcv 1.x version, which is deprecated.
    If this function is no needed, it's good to remove this.

    Args:
        params (list[torch.nn.Parameter]): List of parameters or buffers
            of a model.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params, world_size, bucket_size_mb)
    else:
        for tensor in params:
            dist.all_reduce(tensor.div_(world_size))


def _allreduce_coalesced(tensors: torch.Tensor, world_size: int, bucket_size_mb: int = -1) -> None:
    """It's from mmcv 1.x version, which is deprecated."""
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


@HOOKS.register_module()
class CheckpointHookWithValResults(Hook):  # pylint: disable=too-many-instance-attributes
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        sync_buffer (bool): Whether to synchronize buffers in different
            gpus. Default: False.
    """

    def __init__(
        self,
        interval=-1,
        by_epoch=True,
        save_optimizer=True,
        out_dir=None,
        max_keep_ckpts=-1,
        sync_buffer=False,
        **kwargs,
    ) -> None:
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self._best_model_weight: Optional[Path] = None

    def before_run(self, runner):
        """Set output directopy if not set."""
        if not self.out_dir:
            self.out_dir = runner.work_dir

    def after_train_epoch(self, runner):
        """Checkpoint stuffs after train epoch."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return

        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        save_ema_model = hasattr(runner, "save_ema_model") and runner.save_ema_model
        if save_ema_model:
            backup_model = runner.model
            runner.model = runner.ema_model
        if getattr(runner, "save_ckpt", False):
            runner.logger.info(f"Saving best checkpoint at {runner.epoch + 1} epochs")
            self._save_best_checkpoint(runner)
            runner.save_ckpt = False

        self._save_latest_checkpoint(runner)

        if save_ema_model:
            runner.model = backup_model
            runner.save_ema_model = False

    @master_only
    def _save_best_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        if self._best_model_weight is not None:  # remove previous best model weight
            prev_model_weight = self.out_dir / self._best_model_weight
            if prev_model_weight.exists():
                prev_model_weight.unlink()

        if self.by_epoch:
            weight_name = f"best_epoch_{runner.epoch + 1}.pth"
        else:
            weight_name = f"best_iter_{runner.iter + 1}.pth"
        runner.save_checkpoint(self.out_dir, filename_tmpl=weight_name, save_optimizer=self.save_optimizer, **self.args)

        self._best_model_weight = Path(weight_name)
        if runner.meta is not None:
            runner.meta.setdefault("hook_msgs", dict())
            runner.meta["hook_msgs"]["best_ckpt"] = str(self.out_dir / self._best_model_weight)

    @master_only
    def _save_latest_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        if self.by_epoch:
            weight_name_format = "epoch_{}.pth"
            cur_step = runner.epoch + 1
        else:
            weight_name_format = "iter_{}.pth"
            cur_step = runner.iter + 1

        runner.save_checkpoint(
            self.out_dir,
            filename_tmpl=weight_name_format.format(cur_step),
            save_optimizer=self.save_optimizer,
            **self.args,
        )

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            for _step in range(cur_step - self.max_keep_ckpts * self.interval, 0, -self.interval):
                ckpt_path = self.out_dir / Path(weight_name_format.format(_step))
                if ckpt_path.exists():
                    ckpt_path.unlink()

        if runner.meta is not None:
            cur_ckpt_filename = Path(self.args.get("filename_tmpl", weight_name_format.format(cur_step)))
            runner.meta.setdefault("hook_msgs", dict())
            runner.meta["hook_msgs"]["last_ckpt"] = str(self.out_dir / cur_ckpt_filename)

    def after_train_iter(self, runner):
        """Checkpoint stuffs after train iteration."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        if hasattr(runner, "save_ckpt"):
            if runner.save_ckpt:
                runner.logger.info(f"Saving checkpoint at {runner.iter + 1} iterations")
                if self.sync_buffer:
                    allreduce_params(runner.model.buffers())
                self._save_checkpoint(runner)
            runner.save_ckpt = False


@HOOKS.register_module()
class EnsureCorrectBestCheckpointHook(Hook):
    """EnsureCorrectBestCheckpointHook.

    This hook makes sure that the 'best_mAP' checkpoint points properly to the best model, even if the best model is
    created in the last epoch.
    """

    def after_run(self, runner: Runner):
        """Called after train epoch hooks."""
        runner.call_hook("after_train_epoch")


@HOOKS.register_module()
class SaveInitialWeightHook(Hook):
    """Save the initial weights before training."""

    def __init__(self, save_path, file_name: str = "weights.pth", **kwargs):
        self._save_path = save_path
        self._file_name = file_name
        self._args = kwargs

    def before_run(self, runner):
        """Save initial the weights before training."""
        runner.logger.info("Saving weight before training")
        runner.save_checkpoint(
            self._save_path, filename_tmpl=self._file_name, save_optimizer=False, create_symlink=False, **self._args
        )
