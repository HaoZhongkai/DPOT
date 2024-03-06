#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter( self.scale * torch.rand(2, in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(2, in_channels, out_channels, self.modes1, self.modes2))
        # self.weights1 = nn.Parameter( self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,dtype=torch.cfloat))
        # self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        out_real = torch.einsum("bixy,ioxy->boxy", input.real, weights[0]) - torch.einsum("bixy,ioxy->boxy", input.imag, weights[1])
        out_imag = torch.einsum("bixy,ioxy->boxy", input.real, weights[1]) + torch.einsum("bixy,ioxy->boxy", input.imag, weights[0])
        return torch.view_as_complex(torch.stack([out_real, out_imag],dim=-1))
        # return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x



class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.out_dim = out_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(embed_dim, out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, img_size = 64, n_channels=1,in_timesteps = 10, out_timesteps=1, n_layers=4, patch_size = 1, use_ln=False, multi_channel=True, normalize=False, n_cls=0):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.img_size = img_size
        self.use_ln = use_ln
        self.padding = 2  # pad the domain if input is non-periodic
        self.patch_size = patch_size

        self.normalize = normalize
        self.n_cls = n_cls
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        # self.fc0 = nn.Linear(in_timesteps*n_channels + 2, self.width)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=n_channels * in_timesteps + 2, embed_dim=in_timesteps * n_channels * patch_size + 2, out_dim=width)

        # up sampler size, should be more efficient
        # self.out_layer = nn.Sequential(
        #     nn.Conv2d(in_channels=width, out_channels=n_channels * out_timesteps * patch_size, kernel_size=1, stride=1),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(in_channels=n_channels * out_timesteps * patch_size, out_channels= n_channels * out_timesteps, kernel_size=patch_size, stride=patch_size),
        #     nn.GELU(),
        #     nn.Conv2d(in_channels= n_channels * out_timesteps, out_channels= n_channels * out_timesteps, kernel_size=1, stride=1)
        # )

        # test code v1
        # self.out_layer = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=width, out_channels=width, kernel_size=patch_size, stride=patch_size),
        #     nn.GELU(),
        #     nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, stride=1),
        #     nn.GELU(),
        #     nn.Conv2d(in_channels=width, out_channels=n_channels * out_timesteps, kernel_size=1,stride=1)
        # )

        self.spectral_convs = nn.ModuleList([SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])
        self.convs = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        # self.ln_layers = nn.ModuleList([nn.LayerNorm([ self.in_shape[0], self.in_shape[1]]) for _ in range(self.n_layers)])
        if self.normalize:
            self.scale_feats = nn.Linear(2 * n_channels, width)
        if self.use_ln:
            self.ln_layers = nn.ModuleList([nn.GroupNorm(4, self.width) for _ in range(self.n_layers)])
        # self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        # self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        # self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        # self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        # self.w0 = nn.Conv2d(self.width, self.width, 1)
        # self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        # self.w3 = nn.Conv2d(self.width, self.width, 1)
        # self.ln0 = nn.LayerNorm([self.width, self.in_shape[0], self.in_shape[1]])
        # self.ln1 = nn.LayerNorm([self.width, self.in_shape[0], self.in_shape[1]])
        # self.ln2 = nn.LayerNorm([self.width, self.in_shape[0], self.in_shape[1]])
        # self.ln3 = nn.LayerNorm([self.width, self.in_shape[0], self.in_shape[1]])

        self.fc1 = nn.Linear(self.width, self.width)
        # self.fc1 = nn.Conv2d(self.width, self.width,kernel_size=1,stride=1)
        self.fc2 = nn.Linear(self.width, n_channels * out_timesteps)
        # self.fc2 = nn.Conv2d(self.width, n_channels * out_timesteps, kernel_size=1,stride=1)

        self.cls_head = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, n_cls)
        )

    def forward(self, x):
        T, C = x.shape[-2], x.shape[-1]
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
            x = (x - mu)/ sigma
            scale_feats = self.scale_feats(torch.cat([mu, sigma],dim=-1)).squeeze(-2).permute(0,3,1,2).contiguous()   # B, 1, 1, C
        else:
            scale_feats = 0.0
        x = x.view(*x.shape[:-2], -1)           #### B, X, Y, T*C
        grid = self.get_grid(x)
        x = torch.cat((x, grid), dim=-1)        #### B, X, Y, T*C +2
        # x = self.fc0(x) + scale_feats
        x = x.permute(0, 3, 1, 2).contiguous()

        # print(x.shape, scale_feats.shape, self.normalize)
        x = self.patch_embed(x) + scale_feats
        # x = x + scale_feats

        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.convs[i](x)
            x = x1 + x2
            x = F.gelu(x)
            if self.use_ln:
                x = self.ln_layers[i](x)


        # x1 = self.conv0(x)
        # x2 = self.w0(x)
        # x = x1 + x2
        # x = F.gelu(x)
        # # x = self.ln0(x)
        #
        # x1 = self.conv1(x)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)
        # # x = self.ln1(x)
        #
        #
        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)
        # # x = self.ln2(x)
        #
        #
        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2
        # # x = F.gelu(x)
        # # x = self.ln3(x)

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        # x = self.fc1(x) # conv


        # classification
        cls_token = x.mean(dim=(2, 3), keepdim=False)
        cls_pred = self.cls_head(cls_token)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x) # mlp

        x = F.gelu(x)
        # x = self.out_layer(x)
        # x = x.permute(0, 2, 3, 1)
        x = self.fc2(x)

        x = x.reshape(*x.shape[:3], self.out_timesteps, C)



        if self.normalize:
            x = x * sigma  + mu

        return x, cls_pred

    # def get_grid(self, data):
    #     if self.n_dim == 1:
    #         grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]))
    #         grid = torch.unsqueeze(grid[0], dim=-1)
    #     elif self.n_dim == 2:
    #         grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]))
    #         grid = torch.stack(grid, dim=-1)
    #     elif self.n_dim == 3:
    #         grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]),torch.linspace(0,1, data.shape[3]))
    #         grid = torch.stack(grid, dim=-1)
    #     elif self.n_dim == 4:
    #         grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]),torch.linspace(0,1, data.shape[3]),torch.linspace(0,1, data.shape[4]))
    #         grid = torch.stack(grid, dim=-1)
    #     else:
    #         raise NotImplementedError
    #     grid = grid.to(data.device)
    #     if self.multi_channel:
    #         grid = grid.unsqueeze(-2)
    #         data = torch.cat([torch.tile(grid.unsqueeze(0), [data.shape[0]] + [1] * self.n_dim + [data.shape[-2], 1]), data],dim=-1)
    #     else:
    #         data = torch.cat([torch.tile(grid.unsqueeze(0), [data.shape[0]] + [1] * self.n_dim + [1]), data], dim=-1)
    #
    #     return data

    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, img_size = 64, n_channels=1, in_timesteps = 10, out_timesteps = 1, n_layers = 4, patch_size = 1, use_ln=False, multi_channel=True, normalize=False, n_cls=0):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.img_size = img_size
        self.use_ln = use_ln
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.normalize = normalize

        self.width = width
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_timesteps * n_channels  + 3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.spectral_convs = nn.ModuleList([SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3) for _ in range(self.n_layers)])
        self.convs = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(self.n_layers)])
        # self.ln_layers = nn.ModuleList([nn.LayerNorm([ self.in_shape[0], self.in_shape[1]]) for _ in range(self.n_layers)])
        if self.normalize:
            self.scale_feats = nn.Linear(2 * n_channels, width)
        if self.use_ln:
            self.ln_layers = nn.ModuleList([nn.GroupNorm(4, self.width) for _ in range(self.n_layers)])

        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, n_channels * out_timesteps)



    def forward(self, x):
        T, C = x.shape[-2], x.shape[-1]
        x = x.view(*x.shape[:-2], -1)  #### B, X, Y, Z, T*C
        grid = self.get_grid(x)
        x = torch.cat((x, grid), dim=-1)  #### B, X, Y, Z, T*C +3
        # x = self.fc0(x) + scale_feats
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # print(x.shape, scale_feats.shape, self.normalize)
        # x = x + scale_feats

        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.convs[i](x)
            x = x1 + x2
            x = F.gelu(x)
            if self.use_ln:
                x = self.ln_layers[i](x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)  # mlp

        x = F.gelu(x)
        x = self.fc2(x)

        x = x.reshape(*x.shape[:4], self.out_timesteps, C)


        return x


    def get_grid(self, x):
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(x.device)


class Interpolate(nn.Module):
    def __init__(self, size, dim):
        super(Interpolate, self).__init__()
        self.size = size
        self.dim = dim

    def forward(self, x):
        if self.dim == 1:
            mode = 'linear'
        elif self.dim == 2:
            mode = 'bilinear'
        elif self.dim == 3:
            mode = 'trilinear'
        else:
            raise ValueError("dim can only be 1, 2 or 3.")

        return F.interpolate(x, size=self.size, mode=mode, align_corners=False)


if __name__ == "__main__":

    model = FNO3d(8,8,8,32, 32,n_channels=3,in_timesteps=10,out_timesteps=1,n_layers=4)

    x = torch.rand(4,20,20,20,10,3)
    print(model(x).shape)