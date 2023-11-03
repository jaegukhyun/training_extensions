"""Memory cache hook for logging and freezing MemCacheHandler."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner

from otx.v2.adapters.datumaro.caching.mem_cache_handler import MemCacheHandlerSingleton


@HOOKS.register_module()
class MemCacheHook(Hook):
    """Memory cache hook for logging and freezing MemCacheHandler."""

    def __init__(self) -> None:
        self.handler = MemCacheHandlerSingleton.get()
        # It is because the first evaluation comes at the very beginning of the training.
        # We don't want to cache validation samples first.
        self.handler.freeze()

    def before_epoch(self, runner: Runner) -> None:
        """Before training, unfreeze the handler."""
        # We want to cache training samples first.
        self.handler.unfreeze()

    def after_epoch(self, runner: Runner) -> None:
        """After epoch. Log the handler statistics.

        To prevent it from skipping the validation samples,
        this hook should have lower priority than CustomEvalHook.
        """
        self.handler.freeze()
        runner.logger.info(f"{self.handler}")