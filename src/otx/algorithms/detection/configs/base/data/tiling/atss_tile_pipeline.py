"""Tiling Pipeline of ATSS model for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

tile_cfg = dict(
    tile_size=400, min_area_ratio=0.9, overlap_ratio=0.2, iou_threshold=0.45, max_per_img=1500, filter_empty_gt=True
)

train_pipeline = [
    dict(type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    dict(type="Resize", scale=(400, 400), keep_ratio=False),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="PackDetInputs",
        meta_keys=[
            "ori_filename",
            "flip_direction",
            "scale_factor",
            "flip",
            "ori_shape",
            "filename",
            "img_shape",
        ],
    ),
]


val_pipeline = [
    dict(type="Resize", scale=(400, 400), keep_ratio=False),
    dict(
        type="PackDetInputs",
        meta_keys=["ori_filename", "scale_factor", "ori_shape", "filename", "img_shape"],
    ),
]
test_pipeline = [
    dict(type="Resize", scale=(400, 400), keep_ratio=False),
    dict(
        type="PackDetInputs",
        meta_keys=["ori_filename", "scale_factor", "ori_shape", "filename", "img_shape"],
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="ImageTilingDataset",
        dataset=dict(
            type="OTXDetDataset",
            pipeline=[
                dict(type="LoadImageFromOTXDataset", enable_memcache=True),
                dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
            ],
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
        ),
        pipeline=train_pipeline,
        **tile_cfg
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ImageTilingDataset",
        dataset=dict(
            type="OTXDetDataset",
            pipeline=[
                dict(type="LoadImageFromOTXDataset", enable_memcache=True),
                dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
            ],
        ),
        pipeline=val_pipeline,
        **tile_cfg
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ImageTilingDataset",
        dataset=dict(
            type="OTXDetDataset",
            test_mode=True,
            pipeline=[dict(type="LoadImageFromOTXDataset")],
        ),
        pipeline=test_pipeline,
        **tile_cfg
    ),
)

val_evaluator = dict(
    type="OTXDetMetric",
    metric="mAP",
)
test_evaluator = val_evaluator
