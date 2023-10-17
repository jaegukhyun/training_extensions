"""Model Configuration of SSD model for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name

_base_ = [
    "../../../../../recipes/stages/detection/incremental.py",
    "../../../../common/adapters/mmcv/configs/backbones/mobilenet_v2_w1.yaml",
    "../../base/models/single_stage_detector.py",
]

__width_mult = 1.0

model = dict(
    type="CustomSingleStageDetector",
    data_preprocessor=dict(
        type="DetDataPreprocessor", mean=[0, 0, 0], std=[255, 255, 255], bgr_to_rgb=True, pad_size_divisor=32
    ),
    bbox_head=dict(
        type="CustomSSDHead",
        num_classes=80,
        in_channels=(int(__width_mult * 96), int(__width_mult * 320)),
        use_depthwise=True,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
        loss_balancing=False,
        anchor_generator=dict(
            type="SSDAnchorGeneratorClustered",
            strides=(16, 32),
            reclustering_anchors=True,
            widths=[
                [
                    38.641007923271076,
                    92.49516032784699,
                    271.4234764938237,
                    141.53469410876247,
                ],
                [
                    206.04136086566515,
                    386.6542727907841,
                    716.9892752215089,
                    453.75609561761405,
                    788.4629155558277,
                ],
            ],
            heights=[
                [
                    48.9243877087132,
                    147.73088476194903,
                    158.23569788707474,
                    324.14510379107367,
                ],
                [
                    587.6216059488938,
                    381.60024152086544,
                    323.5988913027747,
                    702.7486097568518,
                    741.4865860938451,
                ],
            ],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2),
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            pos_iou_thr=0.4,
            neg_iou_thr=0.4,
        ),
        use_giou=False,
        use_focal=False,
    ),
    backbone=dict(
        out_indices=(
            4,
            5,
        )
    ),
)

load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions\
/models/object_detection/v2/mobilenet_v2-2s_ssd-992x736.pth"

fp16 = dict(loss_scale=512.0)
ignore = False
