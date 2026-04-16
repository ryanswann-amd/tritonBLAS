"""Two Tower embedding model for tritonBLAS kernel selection.

Ported from GemmKernelSelection/Embedding/two_tower/model/model.py.
Adapted API for tritonBLAS: dict-based categorical features, L2 normalization,
and similarity scoring.

QueryEncoder: MLP tower for GEMM problem features (M, N, K, dtype, etc.)
DocumentEncoder: MLP tower for kernel config features (BLOCK_M, BLOCK_N, etc.)
TwoTower: Wraps both encoders and computes similarity via dot product.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .mlp import MLP
from .resnet import ResNet


class QueryEncoder(nn.Module):
    """Encoder for GEMM problem (query) features.

    Maps raw problem features (M, N, K, dtype flags, etc.) to a
    fixed-size L2-normalized embedding vector.

    Args:
        in_dim: Number of input features.
        hidden_sizes: List of hidden layer widths.
        emb_dim: Output embedding dimension.
        dropout_rate: Dropout probability (0 = no dropout).
        use_norm: If True, use BatchNorm in the MLP backbone.
        use_resnet: If True, use ResNet backbone instead of plain MLP.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: List[int],
        emb_dim: int,
        dropout_rate: float = 0.0,
        use_norm: bool = True,
        use_resnet: bool = False,
    ):
        super().__init__()

        if use_resnet:
            self.encoder = ResNet(in_dim, hidden_sizes, dropout_rate)
        else:
            self.encoder = MLP(
                [in_dim] + hidden_sizes,
                activation="relu",
                batch_norm=use_norm,
                dropout_rate=dropout_rate,
            )

        self.proj = nn.Linear(hidden_sizes[-1], emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode query features to L2-normalized embedding.

        Args:
            x: (batch, in_dim) tensor of query features.

        Returns:
            (batch, emb_dim) L2-normalized embedding.
        """
        h = self.encoder(x)
        emb = self.proj(h)
        return F.normalize(emb, p=2, dim=-1)


class DocumentEncoder(nn.Module):
    """Encoder for kernel config (document) features.

    Handles both continuous features (numeric config params) and
    categorical features (e.g. matrix_instr_nonkdim) via learned
    embeddings, then maps to a fixed-size L2-normalized embedding.

    Args:
        continuous_dim: Number of continuous input features.
        cat_dims: Dict mapping categorical column name to number of
                  categories, e.g. {'matrix_instr_nonkdim': 2}.
        emb_dim: Output embedding dimension.
        hidden_sizes: List of hidden layer widths.
        dropout_rate: Dropout probability (0 = no dropout).
        use_norm: If True, use BatchNorm in the MLP backbone.
        use_resnet: If True, use ResNet backbone instead of plain MLP.
        cat_emb_dim: Dimension of each categorical embedding (default 8).
    """

    def __init__(
        self,
        continuous_dim: int,
        cat_dims: Dict[str, int],
        emb_dim: int,
        hidden_sizes: List[int],
        dropout_rate: float = 0.0,
        use_norm: bool = True,
        use_resnet: bool = False,
        cat_emb_dim: int = 8,
    ):
        super().__init__()

        # Store ordered cat column names so forward() can iterate deterministically
        self.cat_columns = list(cat_dims.keys())

        # One embedding layer per categorical feature
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(n_cat, cat_emb_dim) for name, n_cat in cat_dims.items()}
        )

        # Total input = continuous features + all categorical embeddings
        n_input = continuous_dim + len(self.embeddings) * cat_emb_dim

        if use_resnet:
            self.encoder = ResNet(n_input, hidden_sizes, dropout_rate)
        else:
            self.encoder = MLP(
                [n_input] + hidden_sizes,
                activation="relu",
                batch_norm=use_norm,
                dropout_rate=dropout_rate,
            )

        self.proj = nn.Linear(hidden_sizes[-1], emb_dim)

    def forward(
        self,
        x_continuous: torch.Tensor,
        x_categorical: Dict[str, torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode document features to L2-normalized embedding.

        Args:
            x_continuous: (batch, continuous_dim) tensor of continuous features.
            x_categorical: Dict mapping column name to (batch,) LongTensor of
                           category indices. If None or empty, only continuous
                           features are used.

        Returns:
            (batch, emb_dim) L2-normalized embedding.
        """
        parts = [x_continuous]

        if x_categorical and len(self.embeddings) > 0:
            for name in self.cat_columns:
                cat_tensor = x_categorical[name]
                emb = self.embeddings[name](cat_tensor)  # (batch, cat_emb_dim)
                parts.append(emb)

        x = torch.cat(parts, dim=1)
        h = self.encoder(x)
        emb = self.proj(h)
        return F.normalize(emb, p=2, dim=-1)


class TwoTower(nn.Module):
    """Two Tower model: maps (query, document) pairs to embedding space.

    Computes similarity between GEMM problems and kernel configs via
    dot product of their L2-normalized embeddings.

    Args:
        query_encoder: QueryEncoder instance.
        doc_encoder: DocumentEncoder instance.
    """

    def __init__(
        self,
        query_encoder: QueryEncoder,
        doc_encoder: DocumentEncoder,
    ):
        super().__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder

    def forward(
        self,
        query_features: torch.Tensor,
        doc_continuous: torch.Tensor,
        doc_categoricals: Dict[str, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute embeddings for query and document.

        Args:
            query_features: (batch, query_in_dim) GEMM problem features.
            doc_continuous: (batch, doc_continuous_dim) kernel config continuous features.
            doc_categoricals: Dict of categorical feature tensors for the document.

        Returns:
            Tuple of (query_embedding, doc_embedding), each (batch, emb_dim) L2-normalized.
        """
        query_emb = self.query_encoder(query_features)
        doc_emb = self.doc_encoder(doc_continuous, doc_categoricals)
        return query_emb, doc_emb

    def similarity(
        self, query_emb: torch.Tensor, doc_emb: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity score between query and document embeddings.

        Since embeddings are L2-normalized, dot product equals cosine similarity
        and lies in [-1, 1].

        Args:
            query_emb: (batch, emb_dim) or (N, emb_dim) query embeddings.
            doc_emb: (batch, emb_dim) or (M, emb_dim) document embeddings.

        Returns:
            (batch,) element-wise dot product if shapes match, or
            (N, M) pairwise similarity matrix if broadcasting.
        """
        if query_emb.dim() == 2 and doc_emb.dim() == 2:
            if query_emb.shape[0] == doc_emb.shape[0]:
                # Element-wise: (batch, emb_dim) x (batch, emb_dim) -> (batch,)
                return (query_emb * doc_emb).sum(dim=-1)
            else:
                # Pairwise: (N, emb_dim) x (M, emb_dim) -> (N, M)
                return query_emb @ doc_emb.t()
        return (query_emb * doc_emb).sum(dim=-1)
