# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from ..utils.utils import get_param, resolve_nn_activation
from .activate import resolve_activate_expand


class MLP(nn.Sequential):
    """Multi-layer perceptron.

    The MLP network is a sequence of linear layers and activation functions. The last layer is a linear layer that
    outputs the desired dimension unless the last activation function is specified.

    It provides additional conveniences:
    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        activation: str = "elu",
        activation_kwargs: dict | None = None,
        last_activation: str | None = None,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            activation: Activation function.
            last_activation: Activation function of the last layer. None results in a linear last layer.
        """
        super().__init__()
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]
        self.activation = activation
        # Create layers sequentially
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        # Resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        if activation_mod is None:
            if activation_kwargs is None:
                activation_mod = resolve_activate_expand(activation, hidden_dims_processed[layer_index + 1], hidden_dims_processed[layer_index + 1])
            else:
                activation_mod = resolve_activate_expand(
                    activation,
                    hidden_dims_processed[layer_index + 1],
                    hidden_dims_processed[layer_index + 1],
                    **activation_kwargs,
                )
            if activation_mod is None:
                raise ValueError(f"Invalid activation function '{activation}'. Valid activations are: {list(resolve_nn_activation('').keys())}")
        layers.append(activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            # Resolve activation functions
            activation_mod = resolve_nn_activation(activation)
            if activation_mod is None:
                if activation_kwargs is None:
                    activation_mod = resolve_activate_expand(
                        activation, hidden_dims_processed[layer_index + 1], hidden_dims_processed[layer_index + 1]
                    )
                else:
                    activation_mod = resolve_activate_expand(
                        activation,
                        hidden_dims_processed[layer_index + 1],
                        hidden_dims_processed[layer_index + 1],
                        **activation_kwargs,
                    )
                if activation_mod is None:
                    raise ValueError(f"Invalid activation function '{activation}'. Valid activations are: {list(resolve_nn_activation('').keys())}")
            layers.append(activation_mod)

        # Add last layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            # Compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # Add a layer to reshape the output to the desired shape
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Add last activation function if specified
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        if (last_activation_mod is None) and (last_activation is not None):
            if activation_kwargs is None:
                last_activation_mod = resolve_activate_expand(
                    last_activation, hidden_dims_processed[layer_index + 1], hidden_dims_processed[layer_index + 1]
                )
            else:
                last_activation_mod = resolve_activate_expand(
                    last_activation,
                    hidden_dims_processed[layer_index + 1],
                    hidden_dims_processed[layer_index + 1],
                    **activation_kwargs,
                )
            if last_activation_mod is None:
                raise ValueError(f"Invalid activation function '{last_activation}'. Valid activations are: {list(resolve_nn_activation('').keys())}")
        if last_activation is not None:
            layers.append(last_activation_mod)

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, res: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the MLP."""
        if res is not None:
            for layer in self:
                x = layer(x) + res
        else:
            for layer in self:
                x = layer(x)
        return x

    def frozen_all_layers(self) -> None:
        """Freeze all layers of the MLP."""
        for param in self.parameters():
            param.requires_grad = False

    def unfrozen_all_layers(self) -> None:
        """Unfreeze all layers of the MLP."""
        for param in self.parameters():
            param.requires_grad = True

    def fozen_layers_until(self, layer_index: list[int]) -> None:
        for idx in layer_index:
            for param in self[idx].parameters():
                param.requires_grad = False

    def get_layer_dim_info(self) -> dict[str, list[int] | int]:
        """Return input/output dims and per-layer info."""
        layer_dims: list = []
        input_dim: int | None = None
        output_dim: int | None = None

        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                in_dim = module.in_features
                out_dim = module.out_features
                layer_dims.append((idx, in_dim, out_dim))
                input_dim = in_dim if input_dim is None else input_dim
                output_dim = out_dim

        hidden_dims = [out_dim for _, _, out_dim in layer_dims[:-1]]

        return {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
        }
