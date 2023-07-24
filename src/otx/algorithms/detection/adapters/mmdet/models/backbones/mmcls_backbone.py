"""Register mmcls's backbones."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mmcls.models.backbones import MobileNetV3
from mmdet.models.builder import BACKBONES

BACKBONES.register_module(MobileNetV3)
