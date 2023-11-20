"""OTX adapters.torch.mmengine.mmdet.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from mmdet.registry import VISUALIZERS

from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmdet.registry import MMDetRegistry
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.entities.task_type import TaskType
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


class MMDetEngine(MMXEngine):
    """The MMDetEngine class is responsible for running inference on pre-trained models."""

    def __init__(self, task: TaskType, work_dir: str | Path | None = None) -> None:
        """Initialize a new instance of the MMDetEngine class.

        Args:
            task (TaskType): Task type of engine.
            work_dir (str | Path, optional): The working directory for the engine. Defaults to None.
        """
        super().__init__(task=task, work_dir=work_dir)
        self.registry = MMDetRegistry()

    def _update_config(self, func_args: dict, **kwargs) -> tuple[Config, bool]:
        config, update_check = super()._update_config(func_args, **kwargs)

        if getattr(config, "val_dataloader", None) and not hasattr(config.val_evaluator, "type"):
            config.val_evaluator = {"type": "OTXDetMetric", "metric": "mAP"}
        if getattr(config, "test_dataloader", None) and not hasattr(config.test_evaluator, "type"):
            config.test_evaluator = {"type": "OTXDetMetric", "metric": "mAP"}

        config.default_hooks.checkpoint.save_best = "pascal_voc/mAP"

        if hasattr(config, "visualizer") and config.visualizer.type not in VISUALIZERS:
            config.visualizer = {
                "type": "DetLocalVisualizer",
                "vis_backends": [{"type": "LocalVisBackend"}, {"type": "TensorboardVisBackend"}],
            }

        max_epochs = getattr(config.train_cfg, "max_epochs", None)
        if max_epochs and hasattr(config, "param_scheduler"):
            for scheduler in config.param_scheduler:
                if hasattr(scheduler, "end") and scheduler.end > max_epochs:
                    scheduler.end = max_epochs
                    if hasattr(scheduler, "begin") and scheduler.begin > scheduler.end:
                        scheduler.begin = scheduler.end
                if hasattr(scheduler, "begin") and scheduler.begin > max_epochs:
                    scheduler.begin = max_epochs - 1
        return config, update_check
