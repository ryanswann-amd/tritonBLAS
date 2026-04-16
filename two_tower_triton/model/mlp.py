"""MLP building block for Two Tower model.

Ported from GemmKernelSelection/Embedding/two_tower/model/mlp.py.
Used by both the query encoder (GEMM shapes) and document encoder (kernel configs).
"""

from itertools import pairwise

from torch import nn

ACTIVATIONS = dict(relu=nn.ReLU, gelu=nn.GELU)


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation, normalization, and dropout.

    Args:
        sizes: List of layer dimensions, e.g. [input_dim, 256, 384, 256, emb_dim].
               Creates len(sizes)-1 linear layers.
        activation: Activation function name ('relu' or 'gelu').
        batch_norm: If True, add BatchNorm1d after each activation.
        dropout_rate: Dropout probability (0 = no dropout).
        layer_norm: If True, add LayerNorm after each activation.
    """

    def __init__(self, sizes, activation="relu", batch_norm=False, dropout_rate=0, layer_norm=False):
        super().__init__()

        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS.keys())}")
        activation_cls = ACTIVATIONS[activation]

        self.mlp = nn.Sequential()

        for in_size, out_size in pairwise(sizes):
            self.mlp.append(nn.Linear(in_size, out_size))
            self.mlp.append(activation_cls())
            if batch_norm:
                self.mlp.append(nn.BatchNorm1d(out_size))
            if layer_norm:
                self.mlp.append(nn.LayerNorm(out_size))
            if dropout_rate:
                self.mlp.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.mlp(x)
