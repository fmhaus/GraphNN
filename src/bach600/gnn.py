import torch
import torch.nn as nn
from typing import Literal

class BatchNorm1dND(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
    
    def forward(self, x):
        B, N, F = x.shape
        return self.norm(x.view(B * N, F)).view(B, N, F)

def _get_norm(norm : Literal["layer", "batch"] | None, dim: int):
    if norm == "layer":
        return nn.LayerNorm(dim)
    elif norm == "batch":
        return BatchNorm1dND(dim)
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

        adjacency = adjacency.to(device=device)

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

        self.normalized_adjacency = torch.sparse_coo_tensor(connections.indices(), 
                                                            values, 
                                                            (self.N, self.N), 
                                                            device=connections.device).coalesce()

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
        
        assert self.normalized_adjacency.device == x.device
        
        if is_batched:
            # Batched sparse mm is not supported, reshape for not sparse
            B, N, F = x.shape
            x = x.permute(1, 0, 2).reshape(N, B*F)

        with torch.amp.autocast(x.device.type, enabled=False):
            # sparse mm doesnt support half
            x = torch.sparse.mm(self.normalized_adjacency, x.to(torch.float32))

        if is_batched:
            x = x.reshape(N, B, F).permute(1, 0, 2)

        x = x @ self.weight 

        if self.biases is not None:
            x = x + self.biases

        return x

class GCNBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, graph: GCNGraph, bias: bool = True,
                  norm : Literal["layer", "batch"] | None = "layer",
                  pre_norm: bool = True,
                  dropout: float | None = 0.0, 
                  activation : Literal["gelu", "relu"] | None = "gelu", 
                  residuals : bool = False):
        
        super().__init__()

        assert not residuals or dim_in == dim_out # can only use residuals with matching in out

        self.gcn_layer = GCNLayer(dim_in, dim_out, graph, bias)
        self.norm = _get_norm(norm, dim_in if pre_norm else dim_out)
        self.pre_norm = pre_norm
        self.activation = _get_activation(activation)
        self.dropout = _get_dropout(dropout)
        self.residuals = residuals
    
    def forward(self, x):
        x_in = x
        
        if self.pre_norm:
            x = self.norm(x)
            
        x = self.gcn_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        if self.residuals:
            x = x_in + x
        
        if not self.pre_norm:
            x = self.norm(x)
            
        return x

class _PoolLayer(nn.Module):
    def __init__(self, n_nodes: int, n_clusters: int, residual: bool):
        super().__init__()
        self.residual = residual
        self.assignment = nn.Parameter(torch.empty(n_nodes, n_clusters))
        nn.init.xavier_uniform_(self.assignment)
        self._saved: torch.Tensor | None = None

    def get_assignment(self) -> torch.Tensor:
        return torch.softmax(self.assignment, dim=-1)  # [N, K]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            self._saved = x
        return torch.einsum('nk,bnf->bkf', self.get_assignment(), x)


class _UnpoolLayer(nn.Module):
    def __init__(self, pool: _PoolLayer):
        super().__init__()
        # Wrap in list so PyTorch doesn't register it as a submodule
        # (assignment parameters are already owned by _PoolLayer)
        self._pool_ref = [pool]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool = self._pool_ref[0]
        x = torch.einsum('nk,bkf->bnf', pool.get_assignment(), x)
        if pool.residual and pool._saved is not None:
            x = x + pool._saved
            pool._saved = None
        return x


class GraphPooling:
    """
    Factory that creates paired pool/unpool nn.Module layers sharing the same
    learned assignment matrix S: [N, K].

    Use pool and unpool directly as layers in a sequential model:

        p = GraphPooling(5000, 128, residual=True)
        model = Model(gcn1, p.pool, transformer, p.unpool, gcn2)

    residual=True adds a skip connection: the pre-pool tensor is added back
    after unpooling (same shape [B, N, F] required).
    """
    def __init__(self, n_nodes: int, n_clusters: int, residual: bool = False):
        self.pool = _PoolLayer(n_nodes, n_clusters, residual)
        self.unpool = _UnpoolLayer(self.pool)

class RBFDistanceBias(nn.Module):
    """
    Maps a precomputed [N, N] normalized distance matrix to a learned [N, N]
    attention bias using radial basis functions.

    bias[i,j] = linear( exp(-widths * (dist[i,j] - centers)^2) )
    """
    def __init__(self, n_kernels: int = 16):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(0.0, 1.0, n_kernels))
        self.log_widths = nn.Parameter(torch.zeros(n_kernels))
        self.linear = nn.Linear(n_kernels, 1, bias=True)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [N, N], normalized to [0, 1]
        d = dist.unsqueeze(-1)                                                   # [N, N, 1]
        rbf = torch.exp(-torch.exp(self.log_widths) * (d - self.centers) ** 2)  # [N, N, K]
        return self.linear(rbf).squeeze(-1)                                      # [N, N]


class TransformerGraph:
    def __init__(self, adjacency: torch.Tensor,
                 positions: torch.Tensor | None = None,
                 device=torch.device("cpu")):
        assert len(adjacency.shape) == 2 and adjacency.shape[0] == adjacency.shape[1]
        N = adjacency.shape[0]

        # densify if sparse
        if adjacency.is_sparse:
            adjacency = adjacency.to_dense()

        # add self-loops
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(1)

        # additive mask: 0 where connected, -inf where not
        mask = torch.zeros(N, N, dtype=torch.float32, device=device)
        mask[adjacency == 0] = float('-inf')
        self.attn_mask = mask

        # pairwise distance matrix, normalized to [0, 1]
        if positions is not None:
            assert positions.shape == (N, 2)
            pos = positions.to(device=device, dtype=torch.float32)
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, 2]
            dist = torch.norm(diff, dim=-1)              # [N, N]
            self.dist_matrix = dist / dist.max()
        else:
            self.dist_matrix = None

class GraphTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int,
                 graph: TransformerGraph | None = None,
                 ff_hidden_dim: int | None = None,
                 attention_dropout: float | None = 0.0,
                 feature_dropout: float | None = 0.0,
                 activation: Literal["gelu", "relu"] = "gelu",
                 norm: Literal["layer", "batch"] | None = "layer",
                 pre_norm: bool = True,
                 residuals: bool = True,
                 n_rbf_kernels: int = 16,
                 dtype=torch.float32):
        super().__init__()

        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * dim

        self.mha = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=attention_dropout if attention_dropout else 0.0, dtype=dtype, batch_first=True)

        self.norm1 = _get_norm(norm, dim)
        self.norm2 = _get_norm(norm, dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            _get_activation(activation),
            nn.Linear(ff_hidden_dim, dim)
        )

        self.dropout_feature = _get_dropout(feature_dropout)
        self.pre_norm = pre_norm
        self.residuals = residuals

        if graph is not None:
            self.register_buffer("attn_mask", graph.attn_mask)
            if graph.dist_matrix is not None:
                self.register_buffer("dist_matrix", graph.dist_matrix)
                self.rbf_bias = RBFDistanceBias(n_rbf_kernels)
            else:
                self.dist_matrix = None
                self.rbf_bias = None
        else:
            self.attn_mask = None
            self.dist_matrix = None
            self.rbf_bias = None

    def _get_attn_bias(self) -> torch.Tensor | None:
        bias = self.attn_mask  # [N, N] or None
        if self.rbf_bias is not None:
            rbf = self.rbf_bias(self.dist_matrix)  # [N, N]
            bias = rbf if bias is None else bias + rbf
        return bias

    def forward(self, x):
        x_in = x

        if self.pre_norm:
            x = self.norm1(x)

        attn_output, _ = self.mha(x, x, x, attn_mask=self._get_attn_bias())
        x = self.dropout_feature(attn_output)

        if self.residuals:
            x = x_in + x

        if not self.pre_norm:
            x = self.norm1(x)

        x_in = x

        if self.pre_norm:
            x = self.norm2(x)

        ffn_output = self.ffn(x)
        x = self.dropout_feature(ffn_output)

        if self.residuals:
            x = x_in + x

        if not self.pre_norm:
            x = self.norm2(x)

        return x
    
class Model(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        B, N, H, F = x.shape
        x_reshaped = x.reshape(B, N, H*F)
        result = self.layers(x_reshaped)
        return result.reshape(B, N, H, 3)
