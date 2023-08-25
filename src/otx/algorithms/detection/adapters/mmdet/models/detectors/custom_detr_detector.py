"""OTX DETR Class for mmdetection detectors."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.detr import DETR

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.common.utils.task_adapt import map_class_names

logger = get_logger()


@DETECTORS.register_module()
class CustomDETR(DETR):
    """Custom  DETR with task adapt."""

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
                    self.load_state_dict_pre_hook,
                    task_adapt["dst_classes"],  # model_classes
                    task_adapt["src_classes"],  # ckpt_classes
                )
            )

    def load_state_dict_pre_hook(self, model_classes, ckpt_classes, ckpt_dict, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading."""
        logger.info("----------------- CustomDETR.load_state_dict_pre_hook() called")

        model_classes = list(model_classes)
        ckpt_classes = list(ckpt_classes)
        model2ckpt = map_class_names(model_classes, ckpt_classes)
        logger.info(f"{ckpt_classes} -> {model_classes} ({model2ckpt})")

        model_dict = self.state_dict()
        param_names = [
            "bbox_head.fc_cls.weight",
            "bbox_head.fc_cls.bias",
        ]
        for param_name in param_names:
            ckpt_name = param_name
            if param_name not in model_dict or ckpt_name not in ckpt_dict:
                logger.info(f"Skipping weight copy: {ckpt_name}")
                continue

            # Mix weights
            model_param = model_dict[param_name].clone()
            ckpt_param = ckpt_dict[ckpt_name]
            for model_t, ckpt_t in enumerate(model2ckpt):
                if ckpt_t >= 0:
                    model_param[model_t].copy_(ckpt_param[ckpt_t])

            # Replace checkpoint weight by mixed weights
            ckpt_dict[ckpt_name] = model_param
