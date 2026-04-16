"""Training task for the Two Tower kernel selection model.

Ported from GemmKernelSelection/Embedding/two_tower/task/task.py.
Adapted for tritonBLAS: TwoTower.forward takes separate args
(query_features, doc_continuous, doc_categoricals) instead of a tuple.

ToleranceHuberLoss: Custom Huber loss with tolerance binning to reduce
    sensitivity to small efficiency differences.
TwoTowerTrainTask: Wraps TwoTower with the loss and an auxiliary scoring MLP.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..model.model import TwoTower


class ToleranceHuberLoss(nn.Module):
    """Huber loss with tolerance binning on target efficiency.

    Bins target efficiency values to the nearest ``eps_tol`` increment
    before converting to logit space.  This prevents the model from
    chasing tiny efficiency differences that are within measurement noise.

    Args:
        eps_tol: Tolerance bin width (default 0.015).
        clip_eps: Clamp targets away from 0/1 before logit transform.
    """

    def __init__(self, eps_tol: float = 0.015, clip_eps: float = 1e-3):
        super().__init__()
        self.eps_tol = eps_tol
        self.clip_eps = clip_eps

    def tolerance_bin(self, r: torch.Tensor) -> torch.Tensor:
        """Round efficiency to the nearest tolerance bin."""
        return (r / self.eps_tol).round().mul(self.eps_tol).clamp(0.0, 1.0)

    def logit_clip(self, r: torch.Tensor) -> torch.Tensor:
        """Clamp to (clip_eps, 1-clip_eps) then apply logit transform."""
        r = r.clamp(self.clip_eps, 1.0 - self.clip_eps)
        return torch.log(r / (1.0 - r))

    def forward(
        self,
        logits: torch.Tensor,
        scores: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute tolerance-binned Huber loss.

        Args:
            logits: Model output (raw logits), shape (B,).
            scores: Target efficiency values in [0, 1], shape (B,).
            weights: Unused, kept for API compat.

        Returns:
            Scalar loss.
        """
        scores_binned = self.tolerance_bin(scores)
        targets = self.logit_clip(scores_binned)
        loss = F.smooth_l1_loss(logits, targets)
        return loss


class TwoTowerTrainTask(nn.Module):
    """Training wrapper for TwoTower with auxiliary MLP head.

    Computes dot-product similarity between query and document embeddings,
    then augments the score with a small auxiliary MLP that operates on
    five interaction features:
        [dot_product, diff_norm, hadamard_var, query_norm, doc_norm]

    The final loss is computed on (logits + 0.3 * aux_logits) vs targets.

    Args:
        two_tower: TwoTower model instance.
        normalize_embeddings: Unused (embeddings are always L2-normalized
            inside the encoders). Kept for API compat with reference.
    """

    def __init__(
        self, two_tower: TwoTower, normalize_embeddings: bool = False
    ) -> None:
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.two_tower = two_tower

        self.loss_fn = ToleranceHuberLoss(clip_eps=1e-3)

        # Auxiliary MLP on 5 aggregate interaction features
        agg_input_dim = 5  # [dot, diff_norm, hadamard_var, q_norm, d_norm]
        self.aux_tiny_mlp = nn.Sequential(
            nn.Linear(agg_input_dim, 8), nn.ReLU(), nn.Linear(8, 1)
        )

    def forward(
        self,
        gemm_features: torch.Tensor,
        kernel_continuous: torch.Tensor,
        kernel_categoricals: dict = None,
        targets: torch.Tensor = None,
    ):
        """Forward pass: compute loss and raw logits.

        Args:
            gemm_features: (B, query_dim) GEMM problem features.
            kernel_continuous: (B, doc_continuous_dim) kernel config features.
            kernel_categoricals: Dict of categorical tensors for doc encoder.
            targets: (B,) target efficiency values in [0, 1].

        Returns:
            Tuple of (loss, logits) if targets provided.
            logits only if targets is None.
        """
        query_emb, doc_emb = self.two_tower(
            gemm_features, kernel_continuous, kernel_categoricals
        )

        # Main similarity logits via dot product
        logits = (query_emb * doc_emb).sum(dim=1).squeeze()

        # Auxiliary interaction features
        hadamard = query_emb * doc_emb
        diff = query_emb - doc_emb

        dot_feat = hadamard.sum(dim=1, keepdim=True)
        diff_norm = (diff ** 2).sum(dim=1, keepdim=True)
        hadamard_var = hadamard.var(dim=1, keepdim=True)
        q_norm = (query_emb ** 2).sum(dim=1, keepdim=True)
        d_norm = (doc_emb ** 2).sum(dim=1, keepdim=True)

        agg_features = torch.cat(
            [dot_feat, diff_norm, hadamard_var, q_norm, d_norm], dim=1
        )
        aux_logits = self.aux_tiny_mlp(agg_features).squeeze()

        combined_logits = logits + 0.3 * aux_logits

        if targets is not None:
            loss = self.loss_fn(combined_logits, targets)
            return loss, logits

        return combined_logits
