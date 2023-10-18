"""Data Pipeline of SSD model for Detection Task."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

__dataset_type = "OTXDetDataset"
__img_size = (864, 864)
__img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

train_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        load_ann_cfg=dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
        resize_cfg=dict(
            type="Resize",
            scale=__img_size,
            keep_ratio=True,
            downscale_only=True,
        ),  # Resize to intermediate size if org image is bigger
        to_float32=True,
        enable_memcache=True,  # Cache after resizing image & annotations
    ),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.1),
    dict(type="Resize", scale=__img_size, keep_ratio=False),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="PackDetInputs",
        meta_keys=[
            "ori_filename",
            "flip_direction",
            "scale_factor",
            "gt_ann_ids",
            "flip",
            "ignored_labels",
            "ori_shape",
            "filename",
            "img_shape",
            "pad_shape",
        ],
    ),
]
val_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        load_ann_cfg=dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
        resize_cfg=dict(type="Resize", scale=__img_size, keep_ratio=False),
        eval_mode=True,
        enable_memcache=True,  # Cache after resizing image
    ),
    dict(
        type="PackDetInputs",
        meta_keys=["ori_filename", "scale_factor", "ori_shape", "filename", "img_shape", "pad_shape"],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(type="Resize", scale=__img_size, keep_ratio=False),
    dict(
        type="PackDetInputs",
        meta_keys=["ori_filename", "scale_factor", "ori_shape", "filename", "img_shape", "pad_shape"],
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(type="OTXDetDataset", filter_cfg=dict(filter_empty_gt=True, min_size=32), pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(type="OTXDetDataset", test_mode=True, pipeline=val_pipeline),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(type="OTXDetDataset", test_mode=True, pipeline=test_pipeline),
)

val_evaluator = dict(
    type="VOCMetric",
    metric="mAP",
)
test_evaluator = val_evaluator
