from typing import List, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def compute_edges(points:torch.FloatTensor, cutoff:float=1.) -> torch.LongTensor:
    distances = torch.sqrt(torch.sum((points[:,None, :] - points[None,:,:])**2, dim=-1))
    adjacency_matrix = distances <= cutoff
    edges = torch.nonzero(adjacency_matrix)
    return edges


def scatter_add(
    x: torch.Tensor, idx: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    summed_x = tmp.index_add(dim, idx, x)
    return summed_x


def graph_conv_aggregate(messages:torch.Tensor, edges:torch.Tensor) -> torch.Tensor:
    n_nodes = torch.max(edges) + 1
    incoming = edges[:, 0]
    aggregated_messages = scatter_add(messages, incoming, dim_size=n_nodes, dim=0)
    return aggregated_messages


def graph_conv_compute_message(x:torch.Tensor, edges:torch.Tensor) -> torch.Tensor:
    # we assume here that in- and out-degrees are equal, i.e. undirected graphs
    _, degree = torch.unique(edges, return_counts=True)

    # sparse version of GCN D^{-1/2} (A+I) D^{-1/2}
    # (assuming self-connection is already included in edges)
    incoming = edges[:,0]
    outgoing = edges[:,1]
    message = x[incoming]
    degree_normalization = torch.sqrt(degree[incoming]*degree[outgoing])
    message = message/degree_normalization[:, None]
    return message


def graph_conv_compute_update(
        x: torch.Tensor,
        m: torch.Tensor,
        theta: torch.Tensor,
        activation: Callable) -> torch.Tensor:
    y = torch.mm(m, theta)
    y = activation(y)
    return y
