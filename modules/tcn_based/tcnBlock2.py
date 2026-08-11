import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from typing import List, Tuple

class CausalConv1d(nn.Module):
    """
    Phép tích chập nhân quả.

    Input:
        [B, C_in, T]

    Output:
        [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chỉ padding bên trái để bảo đảm causal convolution.
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlockBTF(nn.Module):
    """
    Temporal block sử dụng định dạng [B, T, F].

    Input:
        x: [B, T, F_in]a

    Output:
        y: [B, T, F_out]

    Bên trong block:
        [B, T, F_in]
            -> transpose
        [B, F_in, T]
            -> Conv1d
        [B, F_out, T]
            -> transpose
        [B, T, F_out]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dilation = dilation

        self.conv1 = CausalConv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.norm1 = nn.BatchNorm1d(out_features)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.norm2 = nn.BatchNorm1d(out_features)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Projection cho residual.
        if in_features != out_features:
            self.residual_projection = nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
            )
        else:
            self.residual_projection = nn.Identity()

        self.output_activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Input phải có shape [B, T, F], "
                f"nhưng nhận được {tuple(x.shape)}."
            )

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Block yêu cầu F_in={self.in_features}, "
                f"nhưng input có shape {tuple(x.shape)}."
            )

        if verbose:
            print(
                f"Block dilation={self.dilation}: "
                f"input [B,T,F] = {tuple(x.shape)}"
            )

        # [B, T, F_in] -> [B, F_in, T]
        x_conv = x.transpose(1, 2).contiguous()

        if verbose:
            print(
                f"  Sau transpose:       {tuple(x_conv.shape)}"
            )

        # Residual:
        # [B, F_in, T] -> [B, F_out, T]
        residual = self.residual_projection(x_conv)

        if verbose:
            print(
                f"  Residual projection: {tuple(residual.shape)}"
            )

        # Main branch.
        y = self.conv1(x_conv)
        y = self.norm1(y)
        y = self.activation1(y)
        y = self.dropout1(y)

        if verbose:
            print(
                f"  Sau convolution 1:   {tuple(y.shape)}"
            )

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.activation2(y)
        y = self.dropout2(y)

        if verbose:
            print(
                f"  Sau convolution 2:   {tuple(y.shape)}"
            )

        # Residual addition.
        y = self.output_activation(y + residual)

        # [B, F_out, T] -> [B, T, F_out]
        y = y.transpose(1, 2).contiguous()

        if verbose:
            print(
                f"  Output [B,T,F]:      {tuple(y.shape)}"
            )

        return y

