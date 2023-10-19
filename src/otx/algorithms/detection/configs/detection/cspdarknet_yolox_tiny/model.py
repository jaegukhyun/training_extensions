"""Model configuration of YOLOX Tiny model for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

_base_ = ["../../../../../recipes/stages/detection/incremental.py", "../../base/models/detector.py"]

model = dict(
    type="CustomYOLOX",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(type="CSPDarknet", deepen_factor=0.33, widen_factor=0.375, out_indices=(2, 3, 4)),
    neck=dict(type="YOLOXPAFPN", in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(type="CustomYOLOXHead", num_classes=80, in_channels=96, feat_channels=96),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65), max_per_img=100),
)
load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions\
/models/object_detection/v2/yolox_tiny_8x8.pth"

fp16 = dict(loss_scale=512.0)
ignore = False
