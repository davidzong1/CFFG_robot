# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal

from ...utils.utils import get_param, resolve_nn_activation


class Conv2dBlock(nn.Module):
    """A building block for Conv2d networks: Conv2d + Normalization + Activation.

    This block applies a 2D convolution followed by an optional normalization layer
    and activation function. It serves as the fundamental building block for
    constructing convolutional neural networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = "same",
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        norm: Literal["batch", "group", "layer", "none"] | None = "batch",
        num_groups: int = 8,
        activation: str = "relu",
        activation_kwargs: dict | None = None,
        order: Literal["conv-norm-act", "conv-act-norm"] = "conv-norm-act",
    ) -> None:
        """Initialize the Conv2dBlock.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to all four sides of the input. Defaults to "same"
                which pads to preserve spatial dimensions when stride=1.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input to output channels.
            bias: If True, adds a learnable bias to the output.
            padding_mode: Padding mode (``'zeros'``, ``'reflect'``, ``'replicate'`` or
                ``'circular'``).
            norm: Normalization layer type. One of ``"batch"``, ``"group"``,
                ``"layer"``, ``"none"``.
            num_groups: Number of groups for GroupNorm. Only used when
                ``norm="group"``.
            activation: Activation function name. Supported values include
                ``"relu"``, ``"elu"``, ``"gelu"``, ``"swish"``, ``"tanh"``,
                ``"sigmoid"``, ``"identity"``, etc.
            activation_kwargs: Additional keyword arguments for the activation
                function.
            order: Order of operations. ``"conv-norm-act"`` applies normalization
                before activation; ``"conv-act-norm"`` applies activation before
                normalization.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm
        self.activation_name = activation

        # Resolve padding for "same" mode
        if padding == "same":
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = tuple(k // 2 for k in kernel_size)

        # Build layers in the specified order
        layers = []

        # Convolution
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        layers.append(conv)

        # Normalization
        norm_layer = self._resolve_norm(norm, out_channels, num_groups)
        if norm_layer is not None:
            layers.append(norm_layer)

        # Activation (wrapped in a container for nn.Sequential compatibility)
        act_layer = resolve_nn_activation(activation)
        if act_layer is None:
            raise ValueError(
                f"Invalid activation function '{activation}'. "
                f"Valid activations are: {list(resolve_nn_activation('').keys())}"
            )
        layers.append(act_layer)

        # Reorder if needed
        if order == "conv-act-norm":
            # Move norm after activation: [0]=conv, [1]=norm, [2]=act
            # Become: [0]=conv, [1]=act, [2]=norm
            if norm_layer is not None:
                layers[1], layers[2] = layers[2], layers[1]

        self.layers = nn.Sequential(*layers)

    @staticmethod
    def _resolve_norm(
        norm: str | None,
        out_channels: int,
        num_groups: int,
    ) -> nn.Module | None:
        """Resolve the normalization layer from the name.

        Args:
            norm: Normalization type.
            out_channels: Number of output channels.
            num_groups: Number of groups for GroupNorm.

        Returns:
            The normalization module, or None if no normalization is requested.
        """
        if norm is None or norm == "none":
            return None
        if norm == "batch":
            return nn.BatchNorm2d(out_channels)
        if norm == "group":
            # Ensure num_groups divides out_channels
            actual_groups = min(num_groups, out_channels)
            if out_channels % actual_groups != 0:
                # Find the largest divisor <= num_groups
                actual_groups = max(
                    g for g in range(1, num_groups + 1) if out_channels % g == 0
                )
            return nn.GroupNorm(num_groups=actual_groups, num_channels=out_channels)
        if norm == "layer":
            return nn.GroupNorm(num_groups=1, num_channels=out_channels)
        raise ValueError(
            f"Invalid normalization type '{norm}'. "
            f"Valid types are: 'batch', 'group', 'layer', 'none'."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Conv2dBlock."""
        return self.layers(x)


class Conv2dNet(nn.Sequential):
    """A flexible Conv2d network composed of stacked Conv2dBlock layers.

    The Conv2dNet is a sequence of Conv2dBlock layers that can be configured
    with per-layer parameters. It supports automatic weight initialization,
    layer freezing/unfreezing, and layer dimension introspection.

    Example usage::

        net = Conv2dNet(
            in_channels=3,
            hidden_channels=[32, 64, 128],
            kernel_sizes=[3, 3, 3],
            strides=[1, 2, 2],
            norm="batch",
            activation="relu",
        )
        x = torch.randn(8, 3, 64, 64)
        y = net(x)  # shape: (8, 128, 16, 16)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: tuple[int, ...] | list[int],
        kernel_sizes: int | tuple[int, int] | list[int | tuple[int, int]] = 3,
        strides: int | tuple[int, int] | list[int | tuple[int, int]] = 1,
        paddings: str | int | tuple[int, int] | list[str | int | tuple[int, int]] = "same",
        dilations: int | tuple[int, int] | list[int | tuple[int, int]] = 1,
        groups: int | list[int] = 1,
        bias: bool | list[bool] = True,
        padding_mode: str | list[str] = "zeros",
        norm: Literal["batch", "group", "layer", "none"] | None = "batch",
        norm_num_groups: int | list[int] = 8,
        activation: str = "relu",
        activation_kwargs: dict | None = None,
        block_order: Literal["conv-norm-act", "conv-act-norm"] = "conv-norm-act",
    ) -> None:
        """Initialize the Conv2dNet.

        Args:
            in_channels: Number of channels in the input image.
            hidden_channels: Number of output channels for each conv block.
                The length determines the number of blocks.
            kernel_sizes: Kernel size(s) for each conv block. If a single value
                is provided, it is used for all blocks.
            strides: Stride(s) for each conv block. If a single value is provided,
                it is used for all blocks.
            paddings: Padding(s) for each conv block. If a single value is provided,
                it is used for all blocks. Defaults to ``"same"``.
            dilations: Dilation(s) for each conv block.
            groups: Number of blocked connection groups per block.
            bias: Whether to use bias in each block.
            padding_mode: Padding mode per block.
            norm: Normalization type for all blocks (or per-block via list).
            norm_num_groups: Number of groups for GroupNorm per block.
            activation: Activation function for all blocks.
            activation_kwargs: Additional kwargs for the activation function.
            block_order: Order of operations within each Conv2dBlock.
        """
        super().__init__()

        num_blocks = len(hidden_channels)

        # Broadcast scalar parameters to per-block lists
        kernel_sizes = self._broadcast(kernel_sizes, num_blocks, "kernel_sizes")
        strides = self._broadcast(strides, num_blocks, "strides")
        paddings = self._broadcast(paddings, num_blocks, "paddings")
        dilations = self._broadcast(dilations, num_blocks, "dilations")
        groups = self._broadcast(groups, num_blocks, "groups")
        bias = self._broadcast(bias, num_blocks, "bias")
        padding_mode = self._broadcast(padding_mode, num_blocks, "padding_mode")
        norm_list = self._broadcast(norm, num_blocks, "norm")
        norm_num_groups = self._broadcast(norm_num_groups, num_blocks, "norm_num_groups")

        # Build blocks
        current_channels = in_channels
        for i in range(num_blocks):
            block = Conv2dBlock(
                in_channels=current_channels,
                out_channels=hidden_channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                dilation=dilations[i],
                groups=groups[i],
                bias=bias[i],
                padding_mode=padding_mode[i],
                norm=norm_list[i],
                num_groups=norm_num_groups[i],
                activation=activation,
                activation_kwargs=activation_kwargs,
                order=block_order,
            )
            self.add_module(f"{i}", block)
            current_channels = hidden_channels[i]

        # Store metadata
        self.in_channels = in_channels
        self.out_channels = hidden_channels[-1] if num_blocks > 0 else in_channels
        self.hidden_channels = list(hidden_channels)
        self.num_blocks = num_blocks

    @staticmethod
    def _broadcast(value, num_blocks: int, name: str) -> list:
        """Broadcast a scalar or list to a list of length num_blocks.

        Args:
            value: The value to broadcast.
            num_blocks: The target number of blocks.
            name: The name of the parameter (for error messages).

        Returns:
            A list of length num_blocks.

        Raises:
            ValueError: If the value is a list with incorrect length.
        """
        if isinstance(value, (list, tuple)):
            if len(value) != num_blocks:
                raise ValueError(
                    f"Expected {name} to have length {num_blocks}, but got {len(value)}."
                )
            return list(value)
        return [value] * num_blocks

    def init_weights(
        self,
        weight_init_fn: str = "kaiming_uniform",
        bias_init_value: float = 0.0,
        weight_init_kwargs: dict | None = None,
    ) -> None:
        """Initialize the weights of all Conv2d layers.

        Args:
            weight_init_fn: Weight initialization method. Supported values:
                ``"kaiming_uniform"``, ``"kaiming_normal"``, ``"xavier_uniform"``,
                ``"xavier_normal"``, ``"orthogonal"``.
            bias_init_value: Value to initialize biases with. Defaults to 0.0.
            weight_init_kwargs: Additional keyword arguments for the weight
                initialization function.
        """
        if weight_init_kwargs is None:
            weight_init_kwargs = {}

        init_map = {
            "kaiming_uniform": nn.init.kaiming_uniform_,
            "kaiming_normal": nn.init.kaiming_normal_,
            "xavier_uniform": nn.init.xavier_uniform_,
            "xavier_normal": nn.init.xavier_normal_,
            "orthogonal": nn.init.orthogonal_,
        }

        init_fn = init_map.get(weight_init_fn)
        if init_fn is None:
            raise ValueError(
                f"Invalid weight init function '{weight_init_fn}'. "
                f"Valid options are: {list(init_map.keys())}."
            )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_fn(module.weight, **weight_init_kwargs)
                if module.bias is not None:
                    nn.init.constant_(module.bias, bias_init_value)

    def frozen_all_layers(self) -> None:
        """Freeze all layers of the Conv2dNet."""
        for param in self.parameters():
            param.requires_grad = False

    def unfrozen_all_layers(self) -> None:
        """Unfreeze all layers of the Conv2dNet."""
        for param in self.parameters():
            param.requires_grad = True

    def frozen_layers_until(self, layer_indices: list[int]) -> None:
        """Freeze specific layers by index.

        Args:
            layer_indices: List of layer indices to freeze.
        """
        for idx in layer_indices:
            for param in self[idx].parameters():
                param.requires_grad = False

    def get_layer_dim_info(self) -> dict:
        """Return input/output channel info and per-layer details.

        Returns:
            A dictionary with keys ``"in_channels"``, ``"out_channels"``,
            ``"hidden_channels"``, and ``"num_blocks"``.
        """
        layer_info: list[dict] = []
        for idx, module in enumerate(self):
            if isinstance(module, Conv2dBlock):
                layer_info.append(
                    {
                        "index": idx,
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "kernel_size": module.kernel_size,
                        "stride": module.stride,
                        "norm": module.norm_type,
                        "activation": module.activation_name,
                    }
                )

        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "hidden_channels": self.hidden_channels,
            "num_blocks": self.num_blocks,
            "layers": layer_info,
        }

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the Conv2dNet.

        Args:
            x: Input tensor of shape ``(batch, in_channels, height, width)``.
            residual: Optional residual tensor to add to the output. Must have
                the same shape as the output.

        Returns:
            Output tensor of shape ``(batch, out_channels, h_out, w_out)``.
        """
        for layer in self:
            x = layer(x)
        if residual is not None:
            x = x + residual
        return x
