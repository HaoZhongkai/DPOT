#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from einops import rearrange, repeat
from collections import OrderedDict
from models.fno import Interpolate

ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}


class UNet1d(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, theta_dim=0, in_shape = None, out_shape = None, width=32, act = 'gelu'):
        super(UNet1d, self).__init__()

        self.features = width
        self.act = ACTIVATION[act]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.theta_dim = theta_dim
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.reshape_last = (in_shape != None) and (in_shape != out_shape)

        self.__name__ = 'UNet'

        self.encoder1 = self._block(in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = self._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = self._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = self._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = self._block(self.features * 8, self.features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            self.features * 16, self.features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            self.features * 8, self.features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            self.features * 4, self.features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            self.features * 2, self.features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(self.features * 2, self.features, name="dec1")

        if in_shape != out_shape:
            self.interpolate = Interpolate(out_shape, dim=1)

        self.conv = nn.Conv1d(
            in_channels=self.features, out_channels=out_channels, kernel_size=1
        )


    # must be 16*n
    def forward(self, x, theta):

        if self.theta_dim:
            x = torch.cat([x, torch.tile(theta.view([theta.shape[0]] + [1] * (x.ndim-2) + [theta.shape[-1]]),[1] + list(x.shape[1:-1]) + [1])],dim=-1)

        x = x.permute(0, 2, 1)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        x = self.interpolate(dec1) if self.reshape_last else dec1

        x = self.conv(x)
        x = x.permute(0, 2, 1)

        return x


    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh1", self.act),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh2", self.act),
                ]
            )
        )


class UNet2d(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, theta_dim=0, in_shape = None, out_shape = None, width=32, act = 'gelu'):
        super(UNet2d, self).__init__()

        self.features = width
        self.act = ACTIVATION[act]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.theta_dim = theta_dim
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.reshape_last = (in_shape != None) and (in_shape != out_shape)

        self.__name__ = 'UNet'

        self.encoder1 = self._block(in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(self.features * 8, self.features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            self.features * 16, self.features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            self.features * 8, self.features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            self.features * 4, self.features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            self.features * 2, self.features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(self.features * 2, self.features, name="dec1")

        if in_shape != out_shape:
            self.interpolate = Interpolate(out_shape, dim=2)


        self.conv = nn.Conv2d(
            in_channels=self.features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, theta):
        if self.theta_dim:
            x = torch.cat([x, torch.tile(theta.view([theta.shape[0]] + [1] * (x.ndim - 2) + [theta.shape[-1]]), [1] + list(x.shape[1:-1]) + [1])], dim=-1)

        x = x.permute(0, 3, 1, 2)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        x = self.interpolate(dec1) if self.reshape_last else dec1

        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)

        return x

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", self.act),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", self.act),
                ]
            )
        )


class UNet3d(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, theta_dim=0, in_shape = None, out_shape = None, width=32, act = 'gelu'):
        super(UNet3d, self).__init__()

        self.features = width
        self.act = ACTIVATION[act]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.theta_dim = theta_dim
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.reshape_last = (in_shape != None) and (in_shape != out_shape)

        self.__name__ = 'UNet'


        self.encoder1 = self._block(in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = self._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = self._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = self._block(self.features * 8, self.features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            self.features * 16, self.features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            self.features * 8, self.features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            self.features * 4, self.features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            self.features * 2, self.features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(self.features * 2, self.features, name="dec1")

        if in_shape != out_shape:
            self.interpolate = Interpolate(out_shape, dim=3)

        self.conv = nn.Conv3d(
            in_channels=self.features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, theta):
        if self.theta_dim:
            x = torch.cat([x, torch.tile(theta.view([theta.shape[0]] + [1] * (x.ndim - 2) + [theta.shape[-1]]), [1] + list(x.shape[1:-1]) + [1])], dim=-1)

        x = x.permute(0, 4, 1, 2, 3)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        x = self.interpolate(dec1) if self.reshape_last else dec1

        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1)
        return x

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "tanh1", self.act),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "tanh2", self.act),
                ]
            )
        )



class UNet(nn.Module):

    def __init__(self, n_dim, in_channels=3, out_channels=1, in_timesteps=10, out_timesteps=1, multi_channel= False,in_shape = None, out_shape = None, width=32, act = 'gelu', n_cls=1):
        super(UNet, self).__init__()

        self.n_dim = n_dim
        self.features = width
        self.act = ACTIVATION[act]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.multi_channel = multi_channel ### unsqueeze last of not multi-channel
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.reshape_last =(out_shape != None) and (in_shape != out_shape)
        self.n_cls = n_cls

        self.__name__ = 'UNet'

        self.padding = [int(np.ceil(s/16)*16 - s) for s in in_shape]
        if self.n_dim == 1:
            self.pool_layer = nn.MaxPool1d
            self.conv_layer = nn.Conv1d
            self.upconv_layer = nn.ConvTranspose1d
            self.norm_layer = nn.BatchNorm1d
        elif self.n_dim == 2:
            self.pool_layer = nn.MaxPool2d
            self.conv_layer = nn.Conv2d
            self.upconv_layer = nn.ConvTranspose2d
            self.norm_layer = nn.BatchNorm2d
        elif self.n_dim == 3:
            self.pool_layer = nn.MaxPool3d
            self.conv_layer = nn.Conv3d
            self.upconv_layer = nn.ConvTranspose3d
            self.norm_layer = nn.BatchNorm3d


        else:
            raise ValueError

        self.encoder1 = self._block(in_channels * in_timesteps + n_dim, self.features, name="enc1")
        self.pool1 = self.pool_layer(kernel_size=2, stride=2)
        self.encoder2 = self._block(self.features, self.features * 2, name="enc2")
        self.pool2 = self.pool_layer(kernel_size=2, stride=2)
        self.encoder3 = self._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = self.pool_layer(kernel_size=2, stride=2)
        self.encoder4 = self._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = self.pool_layer(kernel_size=2, stride=2)

        self.bottleneck = self._block(self.features * 8, self.features * 16, name="bottleneck")

        self.upconv4 = self.upconv_layer(
            self.features * 16, self.features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = self.upconv_layer(
            self.features * 8, self.features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = self.upconv_layer(
            self.features * 4, self.features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = self.upconv_layer(
            self.features * 2, self.features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(self.features * 2, self.features, name="dec1")

        if in_shape != out_shape:
            self.interpolate = Interpolate(out_shape, dim=self.n_dim)

        self.conv = self.conv_layer(
            in_channels=self.features, out_channels=out_timesteps * out_channels, kernel_size=1
        )

    def get_grid(self, data):
        if self.n_dim == 1:
            grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]))
            grid = torch.unsqueeze(grid[0], dim=-1)
        elif self.n_dim == 2:
            grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]))
            grid = torch.stack(grid, dim=-1)
        elif self.n_dim == 3:
            grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]),torch.linspace(0,1, data.shape[3]))
            grid = torch.stack(grid, dim=-1)
        elif self.n_dim == 4:
            grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]),torch.linspace(0,1, data.shape[3]),torch.linspace(0,1, data.shape[4]))
            grid = torch.stack(grid, dim=-1)
        else:
            raise NotImplementedError
        grid = grid.to(data.device)
        # if self.multi_channel:
        #     grid = grid.unsqueeze(-2)
        #     data = torch.cat([torch.tile(grid.unsqueeze(0), [data.shape[0]] + [1] * self.n_dim + [data.shape[-2], 1]), data],dim=-1)
        # else:
        data = torch.cat([torch.tile(grid.unsqueeze(0), [data.shape[0]] + [1] * self.n_dim + [1]), data], dim=-1)

        return data


    def forward(self, x):
        # if not self.multi_channel:
        #     x = x.unsqueeze(-1)
        x = rearrange(x,'b x y t c -> b x y (t c)')
        x = self.get_grid(x)


        if self.n_dim == 1:
            x = x.permute(0, 2, 1)
            x = F.pad(x, (0, self.padding[0]))
        elif self.n_dim == 2:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, self.padding[-1], 0, self.padding[-2]))
        else:
            x = x.permute(0, 4, 1, 2, 3)
            x = F.pad(x, (0, self.padding[-1], 0, self.padding[-2], 0, self.padding[-3]))

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        if self.reshape_last:
            x = self.interpolate(dec1) if self.reshape_last else dec1
        elif self.n_dim == 1:
            x = dec1[...,:-self.padding[0]]
        elif self.n_dim == 2:
            x = dec1[...,:dec1.shape[-2]-self.padding[-2], :dec1.shape[-1] - self.padding[-1]]
        elif self.n_dim == 3:
            x = dec1[...,:dec1.shape[-3]-self.padding[-3], :dec1.shape[-2]-self.padding[-2],:dec1.shape[-1]-self.padding[-1]]


        x = self.conv(x)

        if self.n_dim == 1:
            x = x.permute(0, 2, 1)
        elif self.n_dim == 2:
            x = x.permute(0, 2, 3, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)

        x = rearrange(x, 'b x y (t c)-> b x y t c', t=self.out_timesteps, c=self.out_channels)
        cls = torch.zeros([x.shape[0], self.n_cls]).to(x.device)

        return x, cls

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        self.conv_layer(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", self.norm_layer(num_features=features)),
                    (name + "tanh1", self.act),
                    (
                        name + "conv2",
                        self.conv_layer(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", self.norm_layer(num_features=features)),
                    (name + "tanh2", self.act),
                ]
            )
        )



if __name__ == "__main__":
    ### 1d
    # unet1d = UNet1d(in_channels=3 + 2, out_channels=5, theta_dim=2, in_shape=[128], out_shape=[64], width =32, act= 'gelu')
    # x = torch.randn(8, 128, 3)
    # theta = torch.randn(8,2)
    # y = unet1d(x, theta) # [8, 64, 5]
    # print(y.shape)
    #
    # unet2d = UNet2d(in_channels=3 + 2, out_channels=5, theta_dim=2, in_shape=[128, 128], out_shape=[64, 64], width=32, act='gelu')
    # x = torch.randn(8, 128, 128, 3)
    # theta = torch.randn(8, 2)
    # y = unet2d(x, theta)  # [8, 64, 64, 5]
    # print(y.shape)
    #
    # unet3d = UNet3d(in_channels=3 + 2, out_channels=5, theta_dim=2, in_shape=[16, 16, 16], out_shape=[4, 4, 4], width=32,act='gelu')
    # x = torch.randn(8, 16, 16, 16, 3)
    # theta = torch.randn(8, 2)
    # y = unet3d(x, theta)  # [8, 4, 4, 4,  5]
    # print(y.shape)
    #
    # unet1d = UNet(n_dim=1, in_channels=3 + 2, out_channels=5, theta_dim=2, in_shape=[128], out_shape=[64], width=32, act='gelu')
    # x = torch.randn(8, 128, 3)
    # theta = torch.randn(8, 2)
    # y = unet1d(x, theta)  # [8, 64, 5]
    # print(y.shape)

    unet2d = UNet(n_dim=2, in_channels=3 + 2, out_channels=5, theta_dim=2, in_shape=[1000, 70], out_shape=[70, 70], width=32,act='gelu')
    x = torch.randn(8, 1000, 70, 3)
    theta = torch.randn(8, 2)
    y = unet2d(x, theta)  # [8, 64, 64, 5]
    print(y.shape)

    # unet3d = UNet(n_dim=3, in_channels=3 + 2, out_channels=5, theta_dim=2, in_shape=[14, 15, 16], out_shape=[14, 15, 16], width=32, act='gelu')
    # x = torch.randn(8, 14, 15, 16, 3)
    # theta = torch.randn(8, 2)
    # y = unet3d(x, theta)  # [8, 14, 15, 16,  5]
    # print(y.shape)