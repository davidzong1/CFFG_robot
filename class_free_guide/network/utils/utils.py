# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import Any


def get_param(param: Any, idx: int) -> Any:
    """Get a parameter for the given index.

    Args:
        param: Parameter or list/tuple of parameters.
        idx: Index to get the parameter for.
    """
    if isinstance(param, (tuple, list)):
        return param[idx]
    else:
        return param


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """Resolves the activation function from the name.

    Args:
        act_name: The name of the activation function.

    Returns:
        The activation function.

    Raises:
        ValueError: If the activation function is not found.
    """
    act_dict = {
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "relu": torch.nn.ReLU(),
        "crelu": torch.nn.CELU(),
        "lrelu": torch.nn.LeakyReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "softplus": torch.nn.Softplus(),
        "gelu": torch.nn.GELU(),
        "swish": torch.nn.SiLU(),
        "mish": torch.nn.Mish(),
        "identity": torch.nn.Identity(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        return None
