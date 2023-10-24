"""Data Pipeline of ATSS model for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

backend_args = None

train_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        load_ann_cfg=dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
        resize_cfg=dict(type="Resize", scale=(1088, 800), keep_ratio=True, downscale_only=True),
        enable_memcache=True,
    ),
    dict(type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    dict(
        type="RandomChoiceResize",
        scales=[(992, 736), (896, 736), (1088, 736), (992, 672), (992, 800)],
        keep_ratio=False,
    ),
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
        resize_cfg=dict(type="Resize", scale=(992, 736), keep_ratio=False),
        eval_mode=True,
        enable_memcache=True,
    ),
    dict(
        type="PackDetInputs",
        meta_keys=["ori_filename", "scale_factor", "ori_shape", "filename", "img_shape", "pad_shape"],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(type="Resize", scale=(992, 736), keep_ratio=False),
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
    type="OTXDetMetric",
    metric="mAP",
)
test_evaluator = val_evaluator
