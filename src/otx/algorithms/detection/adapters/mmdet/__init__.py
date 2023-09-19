"""OTX Adapters - mmdet."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from . import models
from .datasets.dataset import OTXDetDataset

__all__ = ["OTXDetDataset", "models"]
