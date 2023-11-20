"""Collection Pipeline for mmengine task."""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from mmcv.transforms import LoadImageFromFile
from mmengine.registry import TRANSFORMS

if TYPE_CHECKING:
    from mmcv.transforms import BaseTransform

import otx.v2.adapters.datumaro.pipelines.load_image_from_otx_dataset as load_image_base


@TRANSFORMS.register_module()
class LoadImageFromOTXDataset(load_image_base.LoadImageFromOTXDataset):
    """Pipeline element that loads an image from a OTX Dataset on the fly."""


@TRANSFORMS.register_module()
class LoadResizeDataFromOTXDataset(load_image_base.LoadResizeDataFromOTXDataset):
    """Load and resize image & annotation with cache support."""

    def _create_resize_op(self, cfg: dict | None = None) -> BaseTransform | None:
        """Creates resize operation."""
        return TRANSFORMS.build(cfg)


@TRANSFORMS.register_module()
class LoadImageFromNDArray(LoadImageFromFile):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used for prediction.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results
