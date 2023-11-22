from __future__ import annotations

from enum import Enum


class OTXTaskType(str, Enum):
    MULTI_CLASS_CLS = "MULTI_CLASS_CLS"
    DETECTION = "DETECTION"
    DETECTION_SEMI_SL = "DETECTION_SEMI_SL"
