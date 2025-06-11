import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_add(
    x: torch.Tensor, idx: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    """
    Sum over given indices as described above.

    Args:
    - x: input tensor
    - idx: index tensor
    - dim: dimension along which is summed
    - dim_size: size of the dimension along which is summed in the output tensor

    Returns:
    - summed_x: tensor with shape: x.shape[:dim] + (dim_size,) + x.shape[dim+1:]
    """
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    summed_x = tmp.index_add(dim, idx, x)
    return summed_x


def SO3GatedNonlinearity_initialize(self, n_in, lmax):
    self.lmax = lmax
    self.n_in = n_in
    ls = torch.arange(0, lmax + 1)
    nls = 2 * ls + 1
    self.lidx = torch.repeat_interleave(ls, nls)
    self.scaling = nn.Linear(n_in, n_in * (lmax + 1))


def SO3GatedNonlinearity_forward(self, x):
    s0 = x[:, 0, :]
    h = self.scaling(s0).reshape(-1, self.lmax + 1, self.n_in)
    h = h[:, self.lidx]
    y = x * torch.sigmoid(h)
    return y


def RadialFilter_initialize(self, n_radial, n_out, cutoff):
    self.n_radial = n_radial
    self.n_out = n_out

    self.offset = torch.linspace(0, cutoff, n_radial)
    self.widths = torch.FloatTensor(
        torch.abs(self.offset[1] - self.offset[0]) * torch.ones_like(self.offset)
    )
    self.linear = nn.Linear(n_radial, n_out, bias=False)


def RadialFilter_forward(self, distances):
    coeff = -0.5 / torch.pow(self.widths, 2)
    diff = distances - self.offset
    y = torch.exp(coeff * torch.pow(diff, 2))
    y = self.linear(y)
    return y


def SO3Convolution_compute_tensor_product(self, x, Wij, Ylm, outgoing):
    Wij = torch.reshape(Wij, (-1, self.lmax + 1, self.n_features))
    Wij = Wij[:, self.lidx[self.idx_l1m1]]

    xj = x[outgoing[:, None], self.idx_l2m2[None, :], :]

    v = (
            Wij
            * Ylm[:, self.idx_l1m1, None]
            * self.clebsch_gordan[None, :, None]
            * xj
    )
    Mij = scatter_add(v, self.idx_out, dim_size=(self.lmax + 1) ** 2, dim=1)
    return Mij

