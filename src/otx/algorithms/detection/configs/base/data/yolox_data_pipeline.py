"""Data Pipeline of YOLOX variants for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

__img_size = (640, 640)

train_pipeline = [
    dict(type="Mosaic", img_scale=__img_size, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        border=(-__img_size[0] // 2, -__img_size[1] // 2),
    ),
    dict(type="MixUp", img_scale=__img_size, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Resize", scale=__img_size, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
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
        resize_cfg=dict(type="Resize", scale=__img_size, keep_ratio=True),
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
    dict(type="Resize", scale=__img_size, keep_ratio=True),
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
    dataset=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            type="OTXDetDataset",
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(
                    type="LoadResizeDataFromOTXDataset",
                    load_ann_cfg=dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
                    resize_cfg=dict(
                        type="Resize",
                        scale=__img_size,
                        keep_ratio=True,
                        downscale_only=True,
                    ),  # Resize to intermediate size if org image is bigger
                    to_float32=False,
                    enable_memcache=True,  # Cache after resizing image & annotations
                ),
            ],
        ),
        pipeline=train_pipeline,
    ),
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
