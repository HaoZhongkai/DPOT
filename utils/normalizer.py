#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import os
import torch
import numpy as np
import operator
import matplotlib.pyplot as plt
import numpy as np
import torch.special as ts
import contextlib

from scipy import interpolate
from functools import reduce
from utils.utilities import timing



def init_normalizer(type, x1, x2, eps=1e-7):
    x_temp = torch.zeros([2,1,1])
    if type == 'unit':
        normalizer = UnitTransformer(x_temp,eps=eps)
        normalizer.mean = x1
        normalizer.std = x2
    elif type == 'pointunit':
        normalizer = PointWiseUnitTransformer(x_temp,eps=eps)
        normalizer.mean = x1
        normalizer.std = x2
    elif type == 'minmax':
        normalizer = MinMaxTransformer(x_temp,eps=eps)
        normalizer.min = x1
        normalizer.max = x2
    else:
        normalizer = IdentityTransformer(x_temp)

    return normalizer

### only available for unit, minmax
# @timing
def cal_normalizer_efficient(type, data_list, eps=1e-7):
    with torch.no_grad():
        norm_dims = list(range(data_list[0].ndim -1))
        if type == 'unit':
            running_mean1 = data_list[0].mean(dim=norm_dims,keepdim=True)
            running_mean2 = (data_list[0]**2).mean(dim=norm_dims,keepdim=True)
            N = np.prod(data_list[0].shape[:-1])
            for i in range(1, len(data_list)):
                M = np.prod(data_list[i].shape[: -1])
                running_mean1 = (N* running_mean1 + M * data_list[i].mean(dim=norm_dims, keepdim=True))/(N+M)
                running_mean2 = (N *running_mean2 + M * (data_list[i]**2).mean(dim=norm_dims, keepdim=True))/(N+M)
            mean = running_mean1
            std = (running_mean2 - running_mean1**2)**0.5
            normalizer = init_normalizer('unit', mean, std, eps=eps)
        elif type == 'minmax':
            running_min = []
            running_max = []
            for i in range(len(data_list)):
                running_min.append(torch.amin(data_list[i], dim=norm_dims))
                running_max.append(torch.amax(data_list[i], dim=norm_dims))
            min = torch.amin(torch.cat(running_min,dim=0), dim=norm_dims, keepdim=True)
            max = torch.amax(torch.cat(running_max,dim=0), dim=norm_dims, keepdim=True)
            normalizer = init_normalizer('minmax', min, max,  eps=eps)
        elif type == 'none':
            normalizer = IdentityTransformer()
        else:
            raise NotImplementedError
        return normalizer




class IdentityTransformer():
    def __init__(self, X=None, eps=1e-4):
        # self.mean = X.mean(dim=0, keepdim=True)
        # self.std = X.std(dim=0, keepdim=True)
        pass


    def to(self, device):
        # self.mean = self.mean.to(device)
        # self.std = self.std.to(device)
        return self

    def cuda(self):
        # self.mean = self.mean.cuda()
        # self.std = self.std.cuda()
        return self

    def cpu(self):
        # self.mean = self.mean.cpu()
        # self.std = self.std.cpu()
        return self

    def transform(self, x, inverse=False, component='all'):
        return x




'''
    Simple normalization layer
'''
class UnitTransformer():
    def __init__(self, X, eps=1e-3):
        self.mean = X.mean(dim=list(range(X.ndim - 1)), keepdim=True)
        self.std = X.std(dim=list(range(X.ndim - 1)), keepdim=True)
        self.eps = eps      ## eps cannot be too small due to numeric precision


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, X, inverse=True,component='all'):
        if component in ['all' , 'all-reduce']:
            if inverse:
                orig_shape = X.shape
                return (X*(self.std + self.eps) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/(self.std +self.eps)
        else:
            if inverse:
                orig_shape = X.shape
                return (X*(self.std[:,component] + self.eps)+ self.mean[:,component]).view(orig_shape)
            else:
                return (X - self.mean[:,component])/(self.std[:,component] + self.eps)


class MinMaxTransformer():
    def __init__(self, X, eps=1e-4):
        self.min = torch.amin(X, dim=list(range(X.ndim - 1)), keepdim=True)
        self.max = torch.amax(X, dim=list(range(X.ndim - 1)), keepdim=True)
        self.eps = eps

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        return self

    def transform(self, X, inverse=True,component='all'):
        if component in ['all', 'all-reduce']:
            if inverse:
                orig_shape = X.shape
                return (X*(self.max - self.min + self.eps) + self.min).view(orig_shape)
            else:
                return (X-self.min)/(self.max -self.min + self.eps)
        else:
            if inverse:
                orig_shape = X.shape
                return (X*(self.max[:,component] - self.min[:,component] + self.eps)+ self.min[:,component]).view(orig_shape)
            else:
                return (X - self.min[:,component])/(self.max[:,component] - self.min[:, component] + self.eps)


'''
    Simple pointwise normalization layer, all data must contain the same length, used only for FNO datasets
    X: B, [N1,...,Nm], C
'''
class PointWiseUnitTransformer():
    def __init__(self, X, temporal=True, eps=1e-4):
        if temporal:
           self.mean = X.mean(dim=(0, -2), keepdim=True)
           self.std = X.std(dim = (0, -2), keepdim=True)
        else:
            self.mean = X.mean(dim=0, keepdim=True)
            self.std = X.std(dim=0, keepdim=True)
            self.eps = eps


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, X, inverse=True,component='all'):
        if component in ['all' , 'all-reduce']:
            if inverse:
                orig_shape = X.shape
                # X = X.view(-1, self.mean.shape[0],self.mean.shape[1])   ### align shape for flat tensor
                return (X*(self.std + self.eps) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/(self.std + self.eps)
        else:
            if inverse:
                orig_shape = X.shape
                # X = X.view(-1, self.mean.shape[0],self.mean.shape[1])
                return (X*(self.std[...,component] + self.eps) + self.mean[...,component]).view(orig_shape)
            else:
                return (X - self.mean[...,component])/(self.std[...,component] + self.eps)




class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)



### TODO: verify errors
class TorchQuantileTransformer():
    '''
    QuantileTransformer implemented by PyTorch
    '''

    def __init__(
            self,
            output_distribution,
            references_,
            quantiles_,
            device=torch.device('cpu')
    ) -> None:
        self.quantiles_ = torch.Tensor(quantiles_).to(device)
        self.output_distribution = output_distribution
        self._norm_pdf_C = np.sqrt(2 * np.pi)
        self.references_ = torch.Tensor(references_).to(device)
        BOUNDS_THRESHOLD = 1e-7
        self.clip_min = self.norm_ppf(torch.Tensor([BOUNDS_THRESHOLD - np.spacing(1)]))
        self.clip_max = self.norm_ppf(torch.Tensor([1 - (BOUNDS_THRESHOLD - np.spacing(1))]))

    def norm_pdf(self, x):
        return torch.exp(-x ** 2 / 2.0) / self._norm_pdf_C

    @staticmethod
    def norm_cdf(x):
        return ts.ndtr(x)

    @staticmethod
    def norm_ppf(x):
        return ts.ndtri(x)

    def transform_col(self, X_col, quantiles, inverse):
        BOUNDS_THRESHOLD = 1e-7
        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col = self.norm_cdf(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if output_distribution == "normal":
                lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
                upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
            if output_distribution == "uniform":
                lower_bounds_idx = X_col == lower_bound_x
                upper_bounds_idx = X_col == upper_bound_x

        isfinite_mask = ~torch.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        torch_interp = Interp1d()
        X_col_out = X_col.clone()
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            X_col_out[isfinite_mask] = 0.5 * (
                    torch_interp(quantiles, self.references_, X_col_finite)
                    - torch_interp(-torch.flip(quantiles, [0]), -torch.flip(self.references_, [0]), -X_col_finite)
            )
        else:
            X_col_out[isfinite_mask] = torch_interp(self.references_, quantiles, X_col_finite)

        X_col_out[upper_bounds_idx] = upper_bound_y
        X_col_out[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col_out = self.norm_ppf(X_col_out)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    X_col_out = torch.clip(X_col_out, self.clip_min, self.clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let X_col unchanged

        return X_col_out

    def transform(self, X, inverse=True,component='all'):
        X_ = X.reshape(-1, X.shape[-1])
        X_out_ = torch.zeros_like(X_)
        for feature_idx in range(X_.shape[1]):
            # X_out[:, feature_idx] = self.transform_col(X[:, feature_idx], self.quantiles_[:, feature_idx], inverse)
            X_out_[:, feature_idx] = self.transform_col(X_[:, feature_idx], self.quantiles_[:, feature_idx], inverse)
        return X_out_.reshape(X.shape)

    def to(self,device):
        self.quantiles_ = self.quantiles_.to(device)
        self.references_ = self.references_.to(device)
        return self