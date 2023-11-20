"""OTX adapters.torch.mmengine.mmpretrain.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import torch

from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmpretrain.registry import MMPretrainRegistry
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.entities.task_type import TaskType
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


class MMPTEngine(MMXEngine):
    """The MMPretrainEngine class is responsible for running inference on pre-trained models."""

    def __init__(self, task: TaskType, work_dir: str | Path | None = None) -> None:
        """Initialize a new instance of the MMPretrainEngine class.

        Args:
            task (TaskType): Task type of engine.
            work_dir (Optional[Union[str, Path]], optional): The working directory for the engine. Defaults to None.
        """
        super().__init__(task=task, work_dir=work_dir)
        self.registry = MMPretrainRegistry()

    def _update_eval_config(self, evaluator_config: list | dict | None, num_classes: int) -> list | dict | None:
        if evaluator_config is None or not evaluator_config:
            evaluator_config = [{"type": "Accuracy"}]
        if isinstance(evaluator_config, list):
            for metric_config in evaluator_config:
                if isinstance(metric_config, dict) and "topk" in metric_config:
                    metric_config["topk"] = [1] if num_classes < 5 else [1, 5]
        elif isinstance(evaluator_config, dict) and "topk" in evaluator_config:
            evaluator_config["topk"] = [1] if num_classes < 5 else [1, 5]
        return evaluator_config

    def _update_config(self, func_args: dict, **kwargs) -> tuple[Config, bool]:
        config, update_check = super()._update_config(func_args, **kwargs)
        num_classes = -1
        model = config.get("model", {})
        if isinstance(model, torch.nn.Module):
            head = model.head if hasattr(model, "head") else None
            num_classes = head.num_classes if hasattr(head, "num_classes") else -1
        else:
            head = model.get("head", {})
            num_classes = head.get("num_classes", -1)
        for subset in ("val", "test"):
            if f"{subset}_dataloader" in config and config[f"{subset}_dataloader"] is not None:
                evaluator_config = self._get_value_from_config(f"{subset}_evaluator", func_args)
                config[f"{subset}_evaluator"] = self._update_eval_config(
                    evaluator_config=evaluator_config, num_classes=num_classes,
                )

        return config, update_check
