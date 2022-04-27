import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export

# __all__ = ["UNet", "Unet", "unet"]

# https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
class ChannelAttention(nn.Module):
    def __init__(self, submodule, in_planes, out_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.submodule = submodule
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.in_planes = in_planes
        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.GELU(),
                               nn.Conv3d(in_planes // ratio, out_planes, 1, bias=False))
        # self.conv_1 = nn.Conv3d()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("CAT X = ", x.shape, flush=True)
        y = self.submodule(x)
        # print("CAT Y = ", y.shape, flush=True)
        x_av = self.avg_pool(x)
        # print("CAT AVG MID = ", x.shape,  self.in_planes, flush=True)
        # print("FC = ", self.fc, flush=True)
        avg_out = self.fc(x_av)
        # print("CAT AVG = ", avg_out.shape, self.in_planes, flush=True)
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        # print("CA output = ", y.shape, out.shape, avg_out.shape, self.in_planes, flush=True)
        # print("CHANNEL = ", out.shape)
        return y*self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, submodule, in_channels, kernel_size=7, out_channels=None, add_conv_1x1=False):
        super(SpatialAttention, self).__init__()
        self.submodule = submodule
        self.conv_flat = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        # self.conv2 = nn.Conv3d(4, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.pool_layer = nn.MaxPool3d(2)
        self.upscale_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.add_conv_1x1 = add_conv_1x1
        if add_conv_1x1:
            if out_channels is None:
                raise Exception("Out channels needed for conv 1x1")
            self.conv_1x1 = nn.Conv3d(in_channels, out_channels, 1, bias=False)
            self.act = torch.nn.PReLU()

    def forward(self, x):
        # print("#"*30)
        # print("SUB = ", self.submodule, "\n", x.shape, flush=True)
        # print("ORIG = ", x.shape)
        y = self.submodule(x)
        # x_2 = self.conv_flat(x)
        # print("X2 = ", x_2.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print("X = ", x.shape, y.shape, avg_out.shape, max_out.shape, flush=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # print("X2 = ", x.shape, flush=True)
        x = self.conv1(x)
        if x.shape[-1] > y.shape[-1]:
            x = self.pool_layer(x)
        elif x.shape[-1] < y.shape[-1]:
            x = self.upscale_layer(x)
        # print("X3 = ", x.shape, y.shape, flush=True)
        # print("SPATIAL = ", x.shape)
        x = y*self.sigmoid(x)
        if self.add_conv_1x1:
            x = self.conv_1x1(x)
            x = self.act(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_c=32, out_c=32, kernel_size=3, activation=nn.GELU):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(n_c, out_c, kernel_size, padding=kernel_size//2, bias=False)
        self.batchnorm = nn.BatchNorm3d(out_c)
        self.activation = activation()
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size, padding=kernel_size//2, bias=False)
        self.batchnorm2 = nn.BatchNorm3d(out_c)
        self.activation2 = activation()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        return x


# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export

__all__ = ["UNet", "Unet", "unet"]


@export("monai.networks.nets")
@alias("Unet")
class UNet(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout=0.0,
    ) -> None:
        """
        Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
        The residual part uses a convolution to change the input dimensions to match the output dimensions
        if this is necessary but will use nn.Identity if not.
        Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

        Args:
            dimensions: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
            strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
            kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
                its length should equal to dimensions. Defaults to 3.
            up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
                its length should equal to dimensions. Defaults to 3.
            num_res_units: number of residual units. Defaults to 0.
            act: activation type and arguments. Defaults to PReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.

        Note: The acceptable spatial size of input data depends on the parameters of the network,
            to set appropriate spatial size, please check the tutorial for more details:
            https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
            Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
            input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
            the inputs must have spatial dimensions that are all multiples of 2^N.
            Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.

        """
        super().__init__()

        if len(channels) < 2:
            raise ValueError(
                "the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError(
                "the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != dimensions:
                raise ValueError(
                    "the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != dimensions:
                raise ValueError(
                    "the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        # self.ca1 = ChannelAttention(16)
        # self.ca1 = ChannelAttention(16)
        # self.sa1 = SpatialAttention()
        # self.sa2 = SpatialAttention()
        # self.sa3 = SpatialAttention()

        def _create_block(
            inc: int,
            outc: int,
            channels: Sequence[int],
            strides: Sequence[int],
            is_top: bool) -> nn.Sequential:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            print("IN CREATE BLOCK : ", inc, outc, channels, strides, is_top)
            c = channels[0]
            s = strides[0]
            print(c, channels)
            subblock: nn.Module
            if len(channels)>2:
                # continue recursion down
                subblock = _create_block(
                    c, c, channels[1:], strides[1:], False)                
                upc = c * 2
                if len(channels) > len(self.channels)-1:
                    add_spatial = True
                    add_channel = False
                else:
                    add_spatial = False
                    add_channel = False
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                # print("## : CHANNELS = ", c, channels)
                subblock = self._get_bottom_layer(c, channels[1])
                print("CREATED bottom LAYER : ", inc, outc, channels, strides, is_top)
                subblock = ChannelAttention(subblock, in_planes=c, out_planes=channels[1])
                # subblock = ChannelAttention(subblock, in_planes=channels[0])
                upc = c + channels[1]
                add_spatial = False
                add_channel = False

            # create layer in downsampling path
            down = self._get_down_layer(inc, c, s, is_top)
            print("CREATED DOWN LAYER : ", inc, c, s, is_top)
            if add_spatial:
                down = SpatialAttention(down, in_channels=inc)
            if add_channel:
                down = ChannelAttention(down, in_planes=inc, out_planes=c)

            # create layer in upsampling path
            if len(channels)==len(self.channels) and add_spatial:
                up = self._get_up_layer(upc, upc, s, is_top)
            else:
                up = self._get_up_layer(upc, outc, s, is_top)
            print("CREATED UP LAYER : ", upc, outc, s, is_top)
            # print(up, flush=True)
            if add_spatial:
                # print("CHANNELS = ", len(channels), channels[0], channels[1], flush=True)
                print("CH = ",channels, len(channels))
                add_conv_1x1 = len(channels)==len(self.channels)
                up = SpatialAttention(up, in_channels=upc, out_channels=outc, add_conv_1x1=add_conv_1x1)
            if add_channel:
                up = ChannelAttention(up, in_planes=upc, out_planes=outc)

            print("OUT OF CREATE BLOCK : ", inc, outc, channels, strides, is_top)
            return nn.Sequential(down, SkipConnection(subblock), up)

        self.model = _create_block(
            in_channels, out_channels, self.channels, self.strides, True)

    def _get_down_layer(self,
        in_channels: int,
        out_channels: int,
        strides: int,
        is_top: bool) -> nn.Module:
        """ 
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        # print("CREATING DOWN LAYER : ", in_channels, out_channels, strides, is_top)
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        return Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
        )

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        # print("CREATING bottom LAYER : ", in_channels, out_channels)
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        # print("CREATING UP LAYER : ", in_channels, out_channels, strides, is_top)
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_top,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


Unet = unet = UNet
