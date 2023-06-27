"""Data Pipeline of YOLOX_L model for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
__img_size = (640, 640)

train_pipeline = [
    dict(type="Mosaic", img_scale=__img_size, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.5, 1.5),
        border=(-__img_size[0] // 2, -__img_size[1] // 2),
    ),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Resize", img_scale=__img_size, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=114.0),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", size=(416, 416), pad_val=114.0),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

__dataset_type = "CocoDataset"
__data_root = "data/coco/"
__samples_per_gpu = 2

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        # make sure to clean up recipe dataset
        _delete_=True,
        type="MultiImageMixDataset",
        dataset=dict(
            type=__dataset_type,
            ann_file=__data_root + "annotations/instances_train2017.json",
            img_prefix=__data_root + "train2017/",
            pipeline=[
                dict(type="LoadImageFromFile", to_float32=False),
                dict(type="LoadAnnotations", with_bbox=True),
            ],
        ),
        pipeline=train_pipeline,
    ),
    val=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_val2017.json",
        img_prefix=__data_root + "val2017/",
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_val2017.json",
        img_prefix=__data_root + "val2017/",
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
