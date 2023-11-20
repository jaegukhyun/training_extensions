# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
import numpy as np
from mmengine.model import BaseModel
from otx.v2.adapters.torch.mmengine.mmpretrain.engine import MMPTEngine
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.entities.task_type import TaskType
from pytest_mock.plugin import MockerFixture


class TestMMPTEngine:
    def test_init(self, tmp_dir_path: Path) -> None:
        engine = MMPTEngine(work_dir=tmp_dir_path, task=TaskType.CLASSIFICATION)
        assert engine.work_dir == tmp_dir_path

    def test_predict(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.dataset.get_default_pipeline", return_value=[])
        mocker.patch("otx.v2.adapters.torch.mmengine.engine.load_checkpoint", return_value=[])
        engine = MMPTEngine(work_dir=tmp_dir_path, task=TaskType.CLASSIFICATION)

        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._config = Config({})

            def test_step(self, input_batch) -> None:
                return [{"prediciton": [0.1, 0.2, 0.7]}] * len(input_batch["inputs"])

        outputs = engine.predict(
            model=MockModule(),
            img=[np.ndarray((224, 224, 3))] * 9,
            checkpoint=tmp_dir_path / "weight.pth",
            batch_size=4,
        )
        assert len(outputs) == 9

        mocker.patch("mmpretrain.registry.MODELS.build", return_value=MockModule())
        engine.predict(
            model={"_config": {}},
            img="tests/assets/classification_dataset_class_incremental/2/22.jpg",
        )

        class MockModel(BaseModel):
            def __init__(self) -> None:
                super().__init__()
                self._metainfo = Config({"results": [Config({"task": "Image Caption"})]})

            def forward(self, *args, **kwargs) -> torch.Tensor:
                return {"prediciton": [0.1, 0.2, 0.7]}

        engine.predict(
            model=MockModel(),
            img="tests/assets/classification_dataset_class_incremental/2/22.jpg",
        )

        with pytest.raises(NotImplementedError):
            engine.predict(
                model="./test.pth",
                img="tests/assets/classification_dataset_class_incremental/2/22.jpg",
            )

    def test_export(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mock_super_export = mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.engine.MMXEngine.export")
        mock_super_export.return_value = {"outputs": {"bin": "test.bin", "xml": "test.xml"}}
        engine = MMPTEngine(work_dir=tmp_dir_path, task=TaskType.CLASSIFICATION)

        result = engine.export(model="model", checkpoint="checkpoint")
        assert result == {"outputs": {"bin": "test.bin", "xml": "test.xml"}}
