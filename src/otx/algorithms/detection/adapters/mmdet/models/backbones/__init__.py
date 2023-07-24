"""Initial file for mmdetection backbones."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from . import imgclsmob, mmcls_backbone
from .mmov_backbone import MMOVBackbone

__all__ = ["imgclsmob", "MMOVBackbone", "mmcls_backbone"]
