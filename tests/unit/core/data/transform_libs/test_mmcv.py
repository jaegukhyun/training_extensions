# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of mmcv data transform."""

import numpy as np
import pytest
from mmcv.transforms.builder import TRANSFORMS
from otx.core.config.data import SubsetConfig
from otx.core.data.entity.base import ImageInfo, OTXDataEntity
from otx.core.data.transform_libs.mmcv import LoadImageFromFile, MMCVTransformLib
from otx.core.types.transformer_libs import TransformLibType


class TestLoadImageFromFile:
    def test_transform(self) -> None:
        transform = LoadImageFromFile()
        data_entity = OTXDataEntity(
            np.ndarray((224, 224, 3)),
            ImageInfo(0, (224, 224, 3), (224, 224, 3), (0, 0, 0), (1.0, 1.0)),
        )
        out = transform.transform(data_entity)
        assert out["img_shape"] == (224, 224)
        assert out["ori_shape"] == (224, 224)


class TestMMCVTransformLib:
    def test_get_builder(self) -> None:
        assert MMCVTransformLib.get_builder() == TRANSFORMS

    def test_check_mandatory_transform(self) -> None:
        class MockNormalTransform:
            def __call__(self, entity: OTXDataEntity) -> OTXDataEntity:
                return entity

        class MockMandatoryTransform:
            def __call__(self, entity: OTXDataEntity) -> OTXDataEntity:
                return entity

        MMCVTransformLib._check_mandatory_transforms(
            transforms=[MockNormalTransform(), MockMandatoryTransform()],
            mandatory_transforms={MockMandatoryTransform},
        )

        with pytest.raises(RuntimeError):
            MMCVTransformLib._check_mandatory_transforms(
                transforms=[MockNormalTransform()],
                mandatory_transforms={MockMandatoryTransform},
            )

    def test_generate(self, mocker) -> None:
        def mock_convert_func(cfg: dict) -> dict:
            return cfg

        config = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transform_lib_type=TransformLibType.MMCV,
            transforms=[{"type": "LoadImageFromFile"}, {"type": "Normalize", "mean": [0, 0, 0], "std": [1, 1, 1]}],
            num_workers=2,
        )

        mocker.patch("otx.core.data.transform_libs.mmcv.convert_conf_to_mmconfig_dict", side_effect=mock_convert_func)
        transforms = MMCVTransformLib.generate(config)
        assert len(transforms) == 2
        assert np.all(transforms[1].mean == np.array([0., 0., 0.]))
