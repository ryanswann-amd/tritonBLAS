"""Unit tests for two_tower_triton.task module.

Tests the training task wrapper (TwoTowerTrainTask) and the custom
ToleranceHuberLoss used during training.
"""

import torch
import pytest

from two_tower_triton.model.model import TwoTower, QueryEncoder, DocumentEncoder
from two_tower_triton.task.task import ToleranceHuberLoss, TwoTowerTrainTask


# ---------------------------------------------------------------------------
# ToleranceHuberLoss tests
# ---------------------------------------------------------------------------


class TestToleranceHuberLoss:
    """Tests for the tolerance-binned Huber loss."""

    def test_tolerance_bin_rounding(self):
        """Values should be rounded to nearest eps_tol increment.

        Note: PyTorch uses banker's rounding (round-half-to-even), so
        we avoid values that land exactly on 0.5 boundaries.
        """
        loss_fn = ToleranceHuberLoss(eps_tol=0.1)
        x = torch.tensor([0.06, 0.14, 0.27, 0.93])
        binned = loss_fn.tolerance_bin(x)
        expected = torch.tensor([0.1, 0.1, 0.3, 0.9])
        assert torch.allclose(binned, expected, atol=1e-6), \
            f"Binned: {binned}, expected: {expected}"

    def test_tolerance_bin_clamping(self):
        """Binned values should be clamped to [0, 1]."""
        loss_fn = ToleranceHuberLoss(eps_tol=0.1)
        x = torch.tensor([-0.1, 1.1])
        binned = loss_fn.tolerance_bin(x)
        assert (binned >= 0.0).all()
        assert (binned <= 1.0).all()

    def test_logit_clip_range(self):
        """Logit transform should work and not produce inf."""
        loss_fn = ToleranceHuberLoss(clip_eps=1e-3)
        x = torch.tensor([0.0, 0.5, 1.0])
        logits = loss_fn.logit_clip(x)
        assert torch.isfinite(logits).all(), f"Non-finite logits: {logits}"

    def test_logit_clip_symmetry(self):
        """logit(0.5) should be ~0."""
        loss_fn = ToleranceHuberLoss()
        x = torch.tensor([0.5])
        logits = loss_fn.logit_clip(x)
        assert torch.allclose(logits, torch.zeros(1), atol=1e-5)

    def test_loss_zero_for_perfect_match(self):
        """Loss should be ~0 when logits match the target logits."""
        loss_fn = ToleranceHuberLoss(eps_tol=0.1, clip_eps=1e-3)
        # Create a score, compute what the binned logit would be, use that as pred
        scores = torch.tensor([0.5, 0.8])
        binned = loss_fn.tolerance_bin(scores)
        target_logits = loss_fn.logit_clip(binned)
        loss = loss_fn(target_logits, scores)
        assert loss.item() < 1e-6, f"Loss should be ~0, got {loss.item()}"

    def test_loss_nonzero_for_mismatch(self):
        """Loss should be significantly positive for bad predictions."""
        loss_fn = ToleranceHuberLoss(eps_tol=0.1, clip_eps=1e-3)
        logits = torch.tensor([5.0, -5.0])  # Very wrong predictions
        scores = torch.tensor([0.5, 0.5])    # Targets near 0 in logit space
        loss = loss_fn(logits, scores)
        assert loss.item() > 0.1, f"Loss should be significant, got {loss.item()}"

    def test_loss_differentiable(self):
        """Loss should be differentiable w.r.t. logits."""
        loss_fn = ToleranceHuberLoss()
        logits = torch.tensor([0.0, 1.0], requires_grad=True)
        scores = torch.tensor([0.5, 0.8])
        loss = loss_fn(logits, scores)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_different_eps_tol(self):
        """Different eps_tol should change binning granularity."""
        fine = ToleranceHuberLoss(eps_tol=0.01)
        coarse = ToleranceHuberLoss(eps_tol=0.1)
        x = torch.tensor([0.123])
        fine_binned = fine.tolerance_bin(x)
        coarse_binned = coarse.tolerance_bin(x)
        # Fine binning should give 0.12, coarse should give 0.1
        assert torch.isclose(fine_binned, torch.tensor([0.12]), atol=1e-6)
        assert torch.isclose(coarse_binned, torch.tensor([0.1]), atol=1e-6)


# ---------------------------------------------------------------------------
# TwoTowerTrainTask tests
# ---------------------------------------------------------------------------


class TestTwoTowerTrainTask:
    """Tests for the training task wrapper."""

    @staticmethod
    def _make_task(query_dim=20, doc_cont_dim=10, emb_dim=32):
        """Create a TwoTowerTrainTask with given dimensions."""
        qe = QueryEncoder(query_dim, [64], emb_dim, 0.0, True, False)
        de = DocumentEncoder(doc_cont_dim, {"cat1": 3}, emb_dim, [64], 0.0, True, False)
        model = TwoTower(qe, de)
        return TwoTowerTrainTask(model)

    def test_forward_with_targets(self):
        """Forward with targets should return (loss, logits)."""
        task = self._make_task()
        q = torch.randn(8, 20)
        d = torch.randn(8, 10)
        cats = {"cat1": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])}
        targets = torch.rand(8).clamp(0.01, 0.99)

        result = task(q, d, cats, targets)
        assert isinstance(result, tuple)
        loss, logits = result
        assert loss.dim() == 0  # scalar
        assert logits.shape == (8,)
        assert torch.isfinite(loss)

    def test_forward_without_targets(self):
        """Forward without targets should return combined_logits only."""
        task = self._make_task()
        q = torch.randn(4, 20)
        d = torch.randn(4, 10)
        cats = {"cat1": torch.tensor([0, 1, 2, 0])}

        result = task(q, d, cats, targets=None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4,)

    def test_loss_decreases_with_training(self):
        """Loss should decrease over a few optimization steps."""
        task = self._make_task(query_dim=10, doc_cont_dim=5, emb_dim=16)
        task.train()
        optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)

        q = torch.randn(16, 10)
        d = torch.randn(16, 5)
        cats = {"cat1": torch.randint(0, 3, (16,))}
        targets = torch.rand(16).clamp(0.05, 0.95)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss, _ = task(q, d, cats, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (comparing first vs last few steps)
        avg_first = sum(losses[:3]) / 3
        avg_last = sum(losses[-3:]) / 3
        assert avg_last < avg_first, \
            f"Loss didn't decrease: first={avg_first:.4f}, last={avg_last:.4f}"

    def test_gradient_flow_through_both_towers(self):
        """Gradients should flow through both query and document encoders."""
        task = self._make_task()
        q = torch.randn(4, 20)
        d = torch.randn(4, 10)
        cats = {"cat1": torch.tensor([0, 1, 2, 0])}
        targets = torch.rand(4).clamp(0.05, 0.95)

        loss, _ = task(q, d, cats, targets)
        loss.backward()

        # Check query encoder gradients
        qe_grads = [p.grad for p in task.two_tower.query_encoder.parameters()
                     if p.grad is not None]
        assert len(qe_grads) > 0, "No gradients in query encoder"

        # Check doc encoder gradients
        de_grads = [p.grad for p in task.two_tower.doc_encoder.parameters()
                     if p.grad is not None]
        assert len(de_grads) > 0, "No gradients in document encoder"

        # Check aux MLP gradients
        aux_grads = [p.grad for p in task.aux_tiny_mlp.parameters()
                      if p.grad is not None]
        assert len(aux_grads) > 0, "No gradients in aux MLP"

    def test_aux_mlp_contribution(self):
        """Aux MLP should contribute to the combined logits (non-zero weight 0.3)."""
        task = self._make_task()
        task.eval()
        q = torch.randn(4, 20)
        d = torch.randn(4, 10)
        cats = {"cat1": torch.tensor([0, 1, 2, 0])}

        # Get combined logits
        combined = task(q, d, cats, targets=None)

        # Get just the dot-product logits
        with torch.no_grad():
            q_emb, d_emb = task.two_tower(q, d, cats)
            dot_logits = (q_emb * d_emb).sum(dim=1)

        # Combined should differ from pure dot product due to aux MLP
        diff = (combined - dot_logits).abs().mean()
        # The difference should be > 0 (aux MLP adds something)
        assert diff > 0, "Aux MLP contributes nothing"

    def test_logits_finite(self):
        """Logits should always be finite for normal inputs."""
        task = self._make_task()
        task.eval()
        torch.manual_seed(42)
        for _ in range(5):
            q = torch.randn(8, 20)
            d = torch.randn(8, 10)
            cats = {"cat1": torch.randint(0, 3, (8,))}
            logits = task(q, d, cats, targets=None)
            assert torch.isfinite(logits).all(), f"Non-finite logits: {logits}"

    def test_batch_size_one(self):
        """Should work with batch_size=1."""
        task = self._make_task()
        task.eval()
        q = torch.randn(1, 20)
        d = torch.randn(1, 10)
        cats = {"cat1": torch.tensor([0])}
        targets = torch.tensor([0.5])

        loss, logits = task(q, d, cats, targets)
        assert torch.isfinite(loss)
        # logits may be 0-dim for batch_size=1 due to squeeze
        assert logits.numel() == 1
