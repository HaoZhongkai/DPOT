#!/usr/bin/env python
#-*- coding:utf-8 _*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}

'''
    A simple MLP class, includes at least 2 layers and n hidden layers
'''
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu',res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Linear(n_input, n_hidden)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)])

        # self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])



    def forward(self, x):
        x = self.act(self.linear_pre(x))
        for i in range(self.n_layers):
            if self.res:
                x = self.act(self.linears[i](x)) + x
            else:
                x = self.act(self.linears[i](x))
            # x = self.act(self.bns[i](self.linears[i](x))) + x

        x = self.linear_post(x)
        return x




#
# class MLPNOScatter(nn.Module):
#     def __init__(self, input_size=2, output_size=3, n_layers=2, n_hidden=64, res=True):
#         super(MLPNOScatter, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.res = res
#
#         self.mlp = MLP(input_size, n_hidden, output_size, n_layers=n_layers, res=self.res)
#         self.__name__ = 'MLP_s'
#
#
#     def forward(self, x, theta):
#
#         feats = torch.cat([x, theta],dim=1)
#
#         feats = self.mlp(feats)
#
#         return feats



class FourierMLP(nn.Module):
    def __init__(self, space_dim=2, theta_dim=1, output_size=3, n_layers=2, n_hidden=64, act='gelu',fourier_dim=0,type='gaussian', sigma=1,res=True):
        super(FourierMLP, self).__init__()
        self.space_dim = space_dim
        self.theta_dim = theta_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.act = act
        self.sigma = sigma
        self.fourier_dim = fourier_dim
        self.type = type
        self.res = res

        if fourier_dim > 0:
            if self.type == 'gaussian':
                self.B = nn.Parameter(sigma *torch.randn([space_dim, fourier_dim]),requires_grad=False)
                freq_dim = fourier_dim
            elif self.type == 'exp':
                freqs = torch.logspace(np.log10(1/2048),np.log10(2048),fourier_dim//space_dim)
                # freqs = 2**torch.arange(-5,5)
                self.B = nn.Parameter(freqs,requires_grad=False)
                freq_dim = len(freqs)*space_dim
            self.theta_mlp = MLP(theta_dim, fourier_dim, fourier_dim, n_layers=3, act=act,res=self.res)
            self.mlp = MLP(2*freq_dim + fourier_dim, n_hidden, output_size, n_layers=n_layers,act=act,res=self.res)
        else:
            self.mlp = MLP(space_dim + theta_dim, n_hidden, output_size, n_layers=n_layers,act=act,res=self.res)

        self.__name__ = 'FourierMLP'


    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            theta = torch.zeros([x.shape[0],1]).to(x.device)   # an ineffective operation
        elif len(args) == 2:
            x, theta = args

        elif len(args) == 3:
            g, u_p, g_u = args
            x = g.ndata['x']
            theta = dgl.broadcast_nodes(g, u_p)

        else:
            raise ValueError
        if self.fourier_dim > 0:
            theta_feats = self.theta_mlp(theta)
            if self.type == 'gaussian':
                x = torch.cat([torch.sin(2*np.pi*x @ self.B), torch.cos(2*np.pi * x @ self.B), theta_feats],dim=1)
            elif self.type == 'exp':
                x = torch.einsum('ij,k->ijk', x, self.B).reshape(x.shape[0], -1)
                x = torch.cat([torch.sin(2*np.pi*x ), torch.cos(2*np.pi * x), theta_feats],dim=1)

        else:
            x = torch.cat([x, theta],dim=1)

        x = self.mlp(x)

        return x

#
# class PreFourierMLP(nn.Module):
#     def __init__(self, space_dim=2, theta_dim=1, output_size=3, n_layers=2, n_hidden=64, act='gelu',fourier_dim=0,type='gaussian', sigma=1, res=True):
#         super(PreFourierMLP, self).__init__()
#         self.space_dim = space_dim
#         self.theta_dim = theta_dim
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.act = act
#         self.sigma = sigma
#         self.fourier_dim = fourier_dim
#         self.type = type
#         self.res = res
#
#         self.rotation_ffn = nn.Sequential(nn.Linear(theta_dim, 128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128, space_dim**2))
#         self.translate_ffn = nn.Sequential(nn.Linear(theta_dim, 128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128, space_dim))
#
#         if fourier_dim > 0:
#             if self.type == 'gaussian':
#                 self.B = nn.Parameter(sigma *torch.randn([space_dim, fourier_dim]),requires_grad=False)
#                 freq_dim = fourier_dim
#             elif self.type == 'exp':
#                 freqs = torch.logspace(np.log10(1/2048),np.log10(2048),fourier_dim//space_dim)
#                 # freqs = 2**torch.arange(-5,5)
#                 self.B = nn.Parameter(freqs,requires_grad=False)
#                 freq_dim = len(freqs)*space_dim
#             self.theta_mlp = MLP(theta_dim, fourier_dim, fourier_dim, n_layers=3, act=act, res=self.res)
#             self.mlp = MLP(2*freq_dim + fourier_dim, n_hidden, output_size, n_layers=n_layers,act=act, res=self.res)
#         else:
#             self.mlp = MLP(space_dim + theta_dim, n_hidden, output_size, n_layers=n_layers,act=act, res=self.res)
#
#         self.__name__ = 'PreFourierMLP'
#
#
#     def forward(self, *args):
#         if len(args) == 1:
#             x = args[0]
#             theta = torch.zeros([x.shape[0],1]).to(x.device)   # an ineffective operation
#         elif len(args) == 2:
#             x, theta = args
#
#         elif len(args) == 3:
#             g, u_p, g_u = args
#             x = g.ndata['x']
#             theta = dgl.broadcast_nodes(g, u_p)
#
#         else:
#             raise ValueError
#
#         ## process pre-rotation and translation
#
#         Q = self.rotation_ffn(theta).reshape(-1, self.space_dim, self.space_dim)
#         b = self.translate_ffn(theta).reshape(-1, self.space_dim)
#         # Q, _ = torch.linalg.qr(Q)
#         x = (Q @ x.unsqueeze(-1)).squeeze() + b
#         # x = x + b
#
#
#         if self.fourier_dim > 0:
#             theta_feats = self.theta_mlp(theta)
#             if self.type == 'gaussian':
#                 x = torch.cat([torch.sin(2*np.pi*x @ self.B), torch.cos(2*np.pi * x @ self.B), theta_feats],dim=1)
#             elif self.type == 'exp':
#                 x = torch.einsum('ij,k->ijk', x, self.B).reshape(x.shape[0], -1)
#                 x = torch.cat([torch.sin(2*np.pi*x ), torch.cos(2*np.pi * x), theta_feats],dim=1)
#
#         else:
#             x = torch.cat([x, theta],dim=1)
#
#         x = self.mlp(x)
#
#         return x



# class MLPNO(nn.Module):
#     def __init__(self, input_size=2, output_size=3, n_layers=2, n_hidden=64):
#         super(MLPNO, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#
#         self.mlp = MLP(input_size, n_hidden, output_size, n_layers=n_layers)
#         self.__name__ = 'MLP'
#
#
#     def forward(self, g, u_p, g_u):
#         u_p_nodes = dgl.broadcast_nodes(g, u_p)
#         feats = torch.cat([g.ndata['x'], u_p_nodes], dim=1)
#         feats = self.mlp(feats)
#
#         return feats










