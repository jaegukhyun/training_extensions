"""OTX SSD Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.registry import MODELS

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.common.utils.task_adapt import map_class_names
from otx.algorithms.detection.adapters.mmdet.models.detectors.loss_dynamics_mixin import DetLossDynamicsTrackingMixin
from otx.algorithms.detection.adapters.mmdet.models.loss_dyns import TrackingLossType

from .l2sp_detector_mixin import L2SPDetectorMixin
from .sam_detector_mixin import SAMDetectorMixin

logger = get_logger()

# TODO: Need to check pylint issues
# pylint: disable=abstract-method, too-many-locals, unused-argument, protected-access


@MODELS.register_module()
class CustomSingleStageDetector(SAMDetectorMixin, DetLossDynamicsTrackingMixin, L2SPDetectorMixin, SingleStageDetector):
    """SAM optimizer & L2SP regularizer enabled custom SSD."""

    TRACKING_LOSS_TYPE = (TrackingLossType.cls, TrackingLossType.bbox)

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Hook for class-sensitive weight loading
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
                    self.load_state_dict_pre_hook,
                    self,  # model
                    task_adapt["dst_classes"],  # model_classes
                    task_adapt["src_classes"],  # chkpt_classes
                )
            )

    @staticmethod
    def load_state_dict_pre_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading."""
        logger.info(f"----------------- CustomSSD.load_state_dict_pre_hook() called w/ prefix: {prefix}")

        # Dst to chkpt mapping index (including BG class)
        model_dict = model.state_dict()
        chkpt_classes = list(chkpt_classes) + ["__BG__"]
        model_classes = list(model_classes) + ["__BG__"]
        num_chkpt_classes = len(chkpt_classes)
        num_model_classes = len(model_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes}")

        # List of class-relevant params
        if prefix + "bbox_head.cls_convs.0.weight" in chkpt_dict:
            param_names = [
                "bbox_head.cls_convs.{}.weight",  # normal
                "bbox_head.cls_convs.{}.bias",
            ]
        elif prefix + "bbox_head.cls_convs.0.0.weight" in chkpt_dict:
            param_names = [
                "bbox_head.cls_convs.{}.3.weight",  # depth-wise: (0)conv -> (1)bn -> (2)act -> (3)conv
                "bbox_head.cls_convs.{}.3.bias",
            ]
        else:
            param_names = []

        # Weight mixing
        for level in range(10):  # For each level (safer inspection loop than 'while True'. Mostly has 2~3 levels)
            level_found = False
            for model_name in param_names:
                model_name = model_name.format(level)
                chkpt_name = prefix + model_name
                if model_name not in model_dict or chkpt_name not in chkpt_dict:
                    logger.info(f"Skipping weight copy: {chkpt_name}")
                    break
                level_found = True

                model_param = model_dict[model_name].clone()
                chkpt_param = chkpt_dict[chkpt_name]

                num_chkpt_anchors = int(chkpt_param.shape[0] / num_chkpt_classes)
                num_model_anchors = int(model_param.shape[0] / num_model_classes)
                num_anchors = min(num_chkpt_anchors, num_model_anchors)
                logger.info(
                    f"Mixing {model_name}: {num_chkpt_anchors}x{num_chkpt_classes} -> "
                    f"{num_model_anchors}x{num_model_classes} anchors"
                )

                for anchor_idx in range(num_anchors):  # For each anchor
                    for model_t, ckpt_t in enumerate(model2chkpt):
                        if ckpt_t >= 0:
                            # Copying only matched weight rows
                            model_param[anchor_idx * num_model_classes + model_t].copy_(
                                chkpt_param[anchor_idx * num_chkpt_classes + ckpt_t]
                            )

                # Replace checkpoint weight by mixed weights
                chkpt_dict[chkpt_name] = model_param
            if not level_found:
                break
