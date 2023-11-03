"""Module for the MMOVBackbone class."""

from typing import Optional, Union

import networkx as nx
from mmpretrain.models.builder import BACKBONES

from otx.v2.adapters.openvino.graph.parsers.cls import cls_base_parser
from otx.v2.adapters.openvino.models.mmov_model import MMOVModel


@BACKBONES.register_module()
class MMOVBackbone(MMOVModel):
    """MMOVBackbone class.

    Args:
        *args: positional arguments.
        **kwargs: keyword arguments.
    """

    @staticmethod
    def parser(graph: nx.MultiDiGraph, **kwargs) -> dict:
        """Parses the input and output of the model.

        Args:
            graph: input graph.
            **kwargs: keyword arguments.

        Returns:
            Dictionary containing input and output of the model.
        """
        output = cls_base_parser(graph, "backbone")
        if output is None:
            raise ValueError("Parser can not determine input and output of model. Please provide them explicitly")
        return output

    def init_weights(self, pretrained: Optional[Union[bool, str]] = None) -> None:
        """Initializes the weights of the model.

        Args:
            pretrained: pretrained weights. Default: None.
        """
        return