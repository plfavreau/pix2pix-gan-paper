import torch
import torch.nn as nn

def cnn_block(
    in_channels, out_channels, kernel_size, stride=1, padding=0, first_layer=False
):
    """
    Creates a CNN block with optional batch normalization
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to both sides of the input
        first_layer: If True, returns only Conv2d without BatchNorm
    """
    if first_layer:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),
        )

def tcnn_block(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    first_layer=False,
):
    """
    Creates a Transposed CNN block with optional batch normalization
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to both sides of the input
        output_padding: Additional size added to one side of the output shape
        first_layer: If True, returns only ConvTranspose2d without BatchNorm
    """
    if first_layer:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),
        ) 