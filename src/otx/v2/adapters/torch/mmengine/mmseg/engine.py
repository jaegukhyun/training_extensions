"""OTX adapters.torch.mmengine.mmseg.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from mmseg.registry import VISUALIZERS

from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmseg.registry import MMSegmentationRegistry
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.entities.task_type import TaskType
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


class MMSegEngine(MMXEngine):
    """The MMSegmentation class is responsible for running inference on pre-trained models."""

    def __init__(self, task: TaskType, work_dir: str | Path | None = None) -> None:
        """Initialize a new instance of the MMPretrainEngine class.

        Args:
            task (TaskType): Task type of engine.
            work_dir (Optional[Union[str, Path]], optional): The working directory for the engine. Defaults to None.
        """
        super().__init__(task=task, work_dir=work_dir)
        self.registry = MMSegmentationRegistry()
        self.visualizer_cfg = {"name": "visualizer", "type": "SegLocalVisualizer"}
        self.evaluator_cfg = {"type": "IoUMetric", "iou_metrics": ["mDice"]}

    def _update_config(self, func_args: dict, **kwargs) -> tuple[Config, bool]:
        """Update the configuration of the runner with the provided arguments.

        Args:
            func_args (dict): The arguments passed to the engine.
            **kwargs: Additional keyword arguments to update the configuration for mmengine.Runner.

        Returns:
            tuple[Config, bool]: Config, True if the configuration was updated, False otherwise.
        """
        config, update_check = super()._update_config(func_args, **kwargs)
        if getattr(config, "val_dataloader", None) and not hasattr(config.val_evaluator, "type"):
            config.val_evaluator = self.evaluator_cfg
            config.val_cfg = {"type": "ValLoop"}
        if getattr(config, "test_dataloader", None) and not hasattr(config.test_evaluator, "type"):
            config.test_evaluator = self.evaluator_cfg
            config.test_cfg = {"type": "TestLoop"}
        if hasattr(config, "visualizer") and config.visualizer.type not in VISUALIZERS:
            config.visualizer = self.visualizer_cfg
        return config, update_check
