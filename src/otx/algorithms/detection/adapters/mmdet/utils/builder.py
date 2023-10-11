"""MMdet model builder."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Optional, Union

import torch
from mmengine.config import Config, ConfigDict
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import load_checkpoint

from otx.algorithms.common.utils.logger import LEVEL

logger = MMLogger.get_current_instance()


def build_detector(
    config: Config,
    train_cfg: Optional[Union[Config, ConfigDict]] = None,
    test_cfg: Optional[Union[Config, ConfigDict]] = None,
    checkpoint: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    cfg_options: Optional[Union[Config, ConfigDict]] = None,
    from_scratch: bool = False,
) -> torch.nn.Module:
    """A builder function for mmdet model.

    Creates a model, based on the configuration in config.
    Note that this function updates 'load_from' attribute of 'config'.
    """

    from mmdet.models import build_detector as origin_build_detector

    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model_cfg = deepcopy(config.model)
    model = origin_build_detector(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    logger.setLevel("WARNING")
    model.init_weights()
    logger.setLevel(LEVEL)
    model = model.to(device)

    checkpoint = checkpoint if checkpoint else config.pop("load_from", None)
    if checkpoint is not None and not from_scratch:
        load_checkpoint(model, checkpoint, map_location=device)
    config.load_from = checkpoint

    return model
