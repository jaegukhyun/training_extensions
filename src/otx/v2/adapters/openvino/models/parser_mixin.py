# type: ignore
# TODO: Need to remove line 1 (ignore mypy) and fix mypy issues
"""Parser mixin modules for otx.v2.adapters.openvino.models."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, Optional, Tuple, Union

import openvino.runtime as ov

from otx.v2.adapters.openvino.graph import Graph
from otx.v2.adapters.openvino.graph.parsers.builder import PARSERS
from otx.v2.api.utils.logger import get_logger

from .ov_model import OVModel

logger = get_logger()


class ParserMixin:
    """ParserMixin class."""

    def parse(
        self,
        model_path_or_model: Union[str, ov.Model],
        weight_path: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        outputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        parser: Optional[Union[str, Callable]] = None,
        **kwargs,
    ) -> Tuple[Union[str, List[str]], Union[str, List[str]]]:
        """Parse function of ParserMixin class."""
        parser = self.parser if parser is None else parser
        if isinstance(parser, str):
            parser = PARSERS.get(parser)

        if not inputs or not outputs:
            graph = OVModel.build_graph(model_path_or_model, weight_path)
            parsed = parser(graph, **kwargs)

            if not isinstance(parsed, dict) or ("inputs" not in parsed and "outputs" not in parsed):
                raise ValueError(f"parser {parser} failed to find inputs and outputs of model. ")
            if isinstance(parsed["inputs"], dict) != isinstance(parsed["outputs"], dict):
                raise ValueError(f"output of parser ({parser}) is not consistent")
            if isinstance(parsed["inputs"], dict) and isinstance(parsed["outputs"], dict):
                if set(parsed["inputs"].keys()) != set(parsed["outputs"].keys()):
                    raise ValueError(
                        f"input keys {parsed['inputs'].keys()} and "
                        f"output keys {parsed['outputs'].keys()} are different.",
                    )

            inputs = inputs if inputs else parsed["inputs"]
            outputs = outputs if outputs else parsed["outputs"]
            logger.info(f"inputs: {inputs}")
            logger.info(f"outputs: {outputs}")

        return inputs, outputs

    @staticmethod
    def parser(
        graph: Graph,
        **kwargs,
    ) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
        """Function parser."""
        return {"inputs": [], "outputs": []}