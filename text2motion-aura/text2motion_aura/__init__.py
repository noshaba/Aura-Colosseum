# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo: text-driven and constrained motion generation model."""

from .model.load_model import AVAILABLE_MODELS, DEFAULT_MODEL, load_model

__all__ = [
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "load_model",
]

# Internal compatibility for upstream NVIDIA configs/checkpoints that may still
# reference the original Python namespace. Public Aura code uses text2motion_aura.
import sys as _sys
_sys.modules.setdefault("kimodo", _sys.modules[__name__])
