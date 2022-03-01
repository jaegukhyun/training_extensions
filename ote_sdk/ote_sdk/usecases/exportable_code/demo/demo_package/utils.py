"""
Utils for demo
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
import json
from pathlib import Path
from typing import Optional

from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_zoo.model_api.models import Model
from openvino.model_zoo.model_api.pipelines import get_user_config

from ote_sdk.entities.label import Domain
from ote_sdk.serialization.label_mapper import LabelSchemaMapper
from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import (
    create_converter,
)
from ote_sdk.usecases.exportable_code.visualizers import AnomalyVisualizer, Visualizer


def get_model_path(path: Optional[Path]) -> Path:
    """
    Get path to model
    """
    model_path = path
    if model_path is None:
        model_path = Path(__file__).parent / "model.xml"
    if not model_path.exists():
        raise IOError("The path to the model was not found.")

    return model_path


def get_parameters(path: Optional[Path]) -> dict:
    """
    Get hyper parameters to creating model
    """
    parameters_path = path
    if parameters_path is None:
        parameters_path = Path(__file__).parent / "config.json"
    if not parameters_path.exists():
        raise IOError("The path to the config was not found.")

    with open(parameters_path, "r", encoding="utf8") as file:
        parameters = json.load(file)

    return parameters


def create_model(
    model_file: Path, config_file: Path, path_to_wrapper: Optional[Path] = None
) -> Model:
    """
    Create model using ModelAPI factory
    """
    plugin_config = get_user_config("CPU", "", None)
    model_adapter = OpenvinoAdapter(
        create_core(), get_model_path(model_file), plugin_config=plugin_config
    )
    parameters = get_parameters(config_file)
    if path_to_wrapper:
        if not path_to_wrapper.exists():
            raise IOError("The path to the model.py was not found.")

        spec = importlib.util.spec_from_file_location("model", path_to_wrapper)  # type: ignore
        model = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(model)
    else:
        print("Using model wrapper from Open Model Zoo ModelAPI")

    # labels for modelAPI wrappers can be empty, because unused in pre- and postprocessing
    parameters["model_parameters"]["labels"] = []
    model = Model.create_model(
        parameters["type_of_model"],
        model_adapter,
        parameters["model_parameters"],
        preload=True,
    )

    return model


def create_output_converter(config_file: Path = None):
    """
    Create annotation converter according to kind of task
    """
    parameters = get_parameters(config_file)
    converter_type = Domain[parameters["converter_type"]]
    labels = LabelSchemaMapper.backward(parameters["model_parameters"]["labels"])
    return create_converter(converter_type, labels)


def create_visualizer(config_file: Path, inference_type: str):
    """
    Create visualizer according to kind of task
    """
    parameters = get_parameters(config_file)
    task_type = parameters["converter_type"]

    if inference_type != "chain" and (
        task_type in ("ANOMALY_CLASSIFICATION", "ANOMALY_SEGMENTATION")
    ):
        return AnomalyVisualizer(window_name="Result")

    return Visualizer(window_name="Result")
