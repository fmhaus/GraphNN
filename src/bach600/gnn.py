import torch
import torch.nn as nn
from typing import Literal

def _get_norm(norm : Literal["layer", "batch"] | None, dim: int):
    if norm == "layer":
        return nn.LayerNorm(dim)
    elif norm == "batch":
        return nn.BatchNorm1d(dim)
    else:
        return nn.Identity()

def _get_activation(activation : Literal["gelu", "relu"] | None):
    if activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
    else:
        return nn.Identity()
    
def _get_dropout(dropout: float | None = 0):
    if dropout is None or dropout <= 0:
        return nn.Identity()
    
    return nn.Dropout(p = dropout)

class GCNGraph:
    def __init__(self, adjacency: torch.Tensor, dtype = torch.float32, device = torch.cpu):

        assert len(adjacency.shape) == 2 and adjacency.shape[0] == adjacency.shape[1]
        self.N = adjacency.shape[0]
        self.dtype = dtype
        
        if not adjacency.is_sparse:
            adjacency = adjacency.to_sparse()

        adjacency.to(device=device)

        indices, values = adjacency.indices(), adjacency.values()

        # create identity indices and values
        id_indices = torch.arange(self.N, device=adjacency.device)
        id_indices = torch.stack((id_indices, id_indices), dim=0)
        id_values = torch.ones(self.N, device=adjacency.device)

        # create connections sparse matrix by concatenating adjacency and identity indices/values
        connections = torch.sparse_coo_tensor(
            torch.cat((indices, id_indices), dim=1),
            torch.cat((values, id_values)),
            (self.N, self.N),
            dtype = dtype,
            device = adjacency.device
        )

        connections = connections.coalesce()

        # Sum to get degrees vector
        degrees = torch.sparse.sum(connections, dim = 1).to_dense()

        # normalize using power -0.5
        deg_inv_sqrt = degrees.pow(-0.5)

        # assume no 0 degrees because self connections are added
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # do matrix multiplications using indices lists
        row, col = connections.indices()
        values = connections.values()
        values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]

        self.normalized_adjacency = torch.sparse_coo_tensor(connections.indices(), values, (self.N, self.N)).coalesce()

class GCNLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, graph: GCNGraph , bias: bool = True):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.N = graph.N    
    
        self.register_buffer("normalized_adjacency", graph.normalized_adjacency)

        self.weight = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty((dim_in, dim_out), dtype=graph.dtype))
        )

        if bias:
            self.biases = nn.Parameter(torch.zeros(dim_out, dtype=graph.dtype))
        else:
            self.register_parameter("biases", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.dim_in 
        assert x.shape[-2] == self.N
        is_batched = len(x.shape) == 3
        
        if is_batched:
            # Batched sparse mm is not supported, reshape for not sparse
            B, N, F = x.shape
            x = x.permute(1, 0, 2).reshape(N, B*F)

        x = torch.sparse.mm(self.normalized_adjacency, x)

        if is_batched:
            x = x.reshape(N, B, F).permute(1, 0, 2)

        x = x @ self.weight 

        if self.biases is not None:
            x = x + self.biases

        return x

class GCNBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, graph: GCNGraph, bias: bool = True,
                  norm : Literal["layer", "batch"] | None = "layer",
                  dropout: float | None = 0.0, 
                  activation : Literal["gelu", "relu"] | None = "gelu", 
                  residuals : bool = False):
        
        super().__init__()

        assert not residuals or dim_in == dim_out # can only use residuals with matching in out

        self.gcn_layer = GCNLayer(dim_in, dim_out, graph, bias)
        self.norm = _get_norm(norm, dim_in)
        self.activation = _get_activation(activation)
        self.dropout = _get_dropout(dropout)
        self.residuals = residuals
    
    def forward(self, x):
        if self.residuals:
            return x + self.get_block_output(x)
        else:
            return self.get_block_output(x)
    
    def get_block_output(self, x):
        x = self.norm(x)
        x = self.gcn_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class TransformerGraph:
    def __init__(self, adjacency: torch.Tensor):
        pass

class GraphTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, 
                 ff_hidden_dim: int | None = None,
                 attention_dropout: float | None = 0.0, 
                 feature_dropout: float | None = 0.0, 
                 activation: Literal["gelu", "relu"] = "relu",
                 norm: Literal["layer", "batch"] | None = "layer",
                 dtype=torch.float32):
        super().__init__()

        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * dim

        dtype = torch.float32

        self.mha = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=attention_dropout, dtype=dtype, batch_first=True)
        
        self.norm1 = _get_norm(norm, dim)
        self.norm2 = _get_norm(norm, dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            _get_activation(activation),
            nn.Linear(ff_hidden_dim, dim)
        )

        self.dropout_feature = _get_dropout(feature_dropout)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout_feature(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout_feature(ffn_output))

        return x
    
class Model(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        B, N, H, F = x.shape
        x_reshaped = x.reshape(B, N, H*F)
        result = self.layers(x_reshaped)
        return result.reshape(B, N, H)
