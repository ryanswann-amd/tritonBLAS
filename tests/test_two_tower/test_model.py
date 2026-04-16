"""Unit tests for two_tower_triton.model module.

Tests model instantiation, forward pass shapes, L2 normalization,
and similarity computation for QueryEncoder, DocumentEncoder, and TwoTower.
"""

import torch
import pytest

from two_tower_triton.model.model import TwoTower, QueryEncoder, DocumentEncoder
from two_tower_triton.model.mlp import MLP
from two_tower_triton.model.resnet import ResNet, ResBlock


# ---------------------------------------------------------------------------
# MLP tests
# ---------------------------------------------------------------------------


class TestMLP:
    """Tests for the MLP building block."""

    def test_output_shape(self):
        """Output shape should match the last size in the sizes list."""
        mlp = MLP([20, 64, 32])
        x = torch.randn(8, 20)
        out = mlp(x)
        assert out.shape == (8, 32)

    def test_with_batchnorm(self):
        """MLP with batch_norm=True should work."""
        mlp = MLP([20, 64, 32], batch_norm=True)
        x = torch.randn(8, 20)
        out = mlp(x)
        assert out.shape == (8, 32)

    def test_with_dropout(self):
        """MLP with dropout should work in eval mode."""
        mlp = MLP([20, 64, 32], dropout_rate=0.1)
        mlp.eval()
        x = torch.randn(8, 20)
        out = mlp(x)
        assert out.shape == (8, 32)

    def test_single_layer(self):
        """MLP with just input->output should work."""
        mlp = MLP([10, 5])
        x = torch.randn(4, 10)
        out = mlp(x)
        assert out.shape == (4, 5)

    def test_invalid_activation(self):
        """Invalid activation should raise ValueError."""
        with pytest.raises(ValueError):
            MLP([10, 5], activation="invalid")

    def test_gelu_activation(self):
        """GELU activation should work."""
        mlp = MLP([10, 5], activation="gelu")
        x = torch.randn(4, 10)
        out = mlp(x)
        assert out.shape == (4, 5)


# ---------------------------------------------------------------------------
# ResNet tests
# ---------------------------------------------------------------------------


class TestResNet:
    """Tests for the ResNet building block."""

    def test_resblock_same_dim(self):
        """ResBlock with same input/output dim (no shortcut projection)."""
        block = ResBlock(64, 64)
        x = torch.randn(8, 64)
        out = block(x)
        assert out.shape == (8, 64)

    def test_resblock_different_dim(self):
        """ResBlock with different dims should use shortcut projection."""
        block = ResBlock(32, 64)
        x = torch.randn(8, 32)
        out = block(x)
        assert out.shape == (8, 64)
        assert block.shortcut is not None

    def test_resnet_output_shape(self):
        """ResNet output shape should match last hidden size."""
        net = ResNet(20, [64, 64])
        x = torch.randn(8, 20)
        out = net(x)
        assert out.shape == (8, 64)

    def test_resnet_varying_hidden(self):
        """ResNet with varying hidden sizes."""
        net = ResNet(20, [64, 128, 64])
        x = torch.randn(4, 20)
        out = net(x)
        assert out.shape == (4, 64)


# ---------------------------------------------------------------------------
# QueryEncoder tests
# ---------------------------------------------------------------------------


class TestQueryEncoder:
    """Tests for the GEMM problem (query) encoder."""

    def test_output_shape(self):
        """Output should have shape (batch, emb_dim)."""
        qe = QueryEncoder(in_dim=20, hidden_sizes=[64, 64], emb_dim=32,
                          dropout_rate=0.0, use_norm=True, use_resnet=False)
        x = torch.randn(8, 20)
        emb = qe(x)
        assert emb.shape == (8, 32)

    def test_l2_normalized(self):
        """Output embeddings should be L2-normalized (unit norm)."""
        qe = QueryEncoder(in_dim=20, hidden_sizes=[64, 64], emb_dim=32,
                          dropout_rate=0.0, use_norm=True, use_resnet=False)
        qe.eval()
        x = torch.randn(8, 20)
        emb = qe(x)
        norms = torch.norm(emb, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5), \
            f"Norms not unit: {norms}"

    def test_with_resnet_backbone(self):
        """QueryEncoder with ResNet backbone should work."""
        qe = QueryEncoder(in_dim=20, hidden_sizes=[64, 64], emb_dim=32,
                          dropout_rate=0.0, use_norm=True, use_resnet=True)
        x = torch.randn(8, 20)
        emb = qe(x)
        assert emb.shape == (8, 32)
        norms = torch.norm(emb, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_single_hidden_layer(self):
        """Should work with a single hidden layer."""
        qe = QueryEncoder(in_dim=10, hidden_sizes=[32], emb_dim=16,
                          dropout_rate=0.0, use_norm=False, use_resnet=False)
        x = torch.randn(4, 10)
        emb = qe(x)
        assert emb.shape == (4, 16)

    def test_deterministic_eval(self):
        """Same input should produce same output in eval mode."""
        qe = QueryEncoder(in_dim=10, hidden_sizes=[32], emb_dim=16,
                          dropout_rate=0.1, use_norm=True, use_resnet=False)
        qe.eval()
        x = torch.randn(4, 10)
        emb1 = qe(x)
        emb2 = qe(x)
        assert torch.allclose(emb1, emb2)


# ---------------------------------------------------------------------------
# DocumentEncoder tests
# ---------------------------------------------------------------------------


class TestDocumentEncoder:
    """Tests for the kernel config (document) encoder."""

    def test_output_shape(self):
        """Output should have shape (batch, emb_dim)."""
        de = DocumentEncoder(
            continuous_dim=10, cat_dims={"cat1": 3}, emb_dim=32,
            hidden_sizes=[64], dropout_rate=0.0, use_norm=True, use_resnet=False,
        )
        x_cont = torch.randn(4, 10)
        x_cat = {"cat1": torch.tensor([0, 1, 2, 0])}
        emb = de(x_cont, x_cat)
        assert emb.shape == (4, 32)

    def test_l2_normalized(self):
        """Output embeddings should be L2-normalized."""
        de = DocumentEncoder(
            continuous_dim=10, cat_dims={"cat1": 3}, emb_dim=32,
            hidden_sizes=[64], dropout_rate=0.0, use_norm=True, use_resnet=False,
        )
        de.eval()
        x_cont = torch.randn(4, 10)
        x_cat = {"cat1": torch.tensor([0, 1, 2, 0])}
        emb = de(x_cont, x_cat)
        norms = torch.norm(emb, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_no_categoricals(self):
        """Should work with empty categorical dict."""
        de = DocumentEncoder(
            continuous_dim=10, cat_dims={}, emb_dim=16,
            hidden_sizes=[32], dropout_rate=0.0, use_norm=False, use_resnet=False,
        )
        x_cont = torch.randn(4, 10)
        emb = de(x_cont, {})
        assert emb.shape == (4, 16)

    def test_none_categoricals(self):
        """Should work with None categorical dict when no cat_dims."""
        de = DocumentEncoder(
            continuous_dim=10, cat_dims={}, emb_dim=16,
            hidden_sizes=[32], dropout_rate=0.0, use_norm=False, use_resnet=False,
        )
        x_cont = torch.randn(4, 10)
        emb = de(x_cont, None)
        assert emb.shape == (4, 16)

    def test_multiple_categoricals(self):
        """Should handle multiple categorical features."""
        de = DocumentEncoder(
            continuous_dim=5, cat_dims={"c1": 4, "c2": 3}, emb_dim=16,
            hidden_sizes=[32], dropout_rate=0.0, use_norm=False, use_resnet=False,
        )
        x_cont = torch.randn(4, 5)
        x_cat = {"c1": torch.tensor([0, 1, 2, 3]), "c2": torch.tensor([0, 1, 2, 0])}
        emb = de(x_cont, x_cat)
        assert emb.shape == (4, 16)

    def test_with_resnet(self):
        """DocumentEncoder with ResNet backbone should work."""
        de = DocumentEncoder(
            continuous_dim=10, cat_dims={"cat1": 3}, emb_dim=32,
            hidden_sizes=[64, 64], dropout_rate=0.0, use_norm=True, use_resnet=True,
        )
        x_cont = torch.randn(8, 10)
        x_cat = {"cat1": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])}
        emb = de(x_cont, x_cat)
        assert emb.shape == (8, 32)

    def test_custom_cat_emb_dim(self):
        """Custom cat_emb_dim should change internal dimensions."""
        de = DocumentEncoder(
            continuous_dim=10, cat_dims={"cat1": 3}, emb_dim=32,
            hidden_sizes=[64], dropout_rate=0.0, use_norm=False,
            use_resnet=False, cat_emb_dim=16,
        )
        x_cont = torch.randn(4, 10)
        x_cat = {"cat1": torch.tensor([0, 1, 2, 0])}
        emb = de(x_cont, x_cat)
        assert emb.shape == (4, 32)


# ---------------------------------------------------------------------------
# TwoTower tests
# ---------------------------------------------------------------------------


class TestTwoTower:
    """Tests for the full Two Tower model."""

    @staticmethod
    def _make_model(query_dim=20, doc_cont_dim=10, emb_dim=32):
        """Create a TwoTower model with given dimensions."""
        qe = QueryEncoder(query_dim, [64], emb_dim, 0.0, True, False)
        de = DocumentEncoder(doc_cont_dim, {"cat1": 3}, emb_dim, [64], 0.0, True, False)
        return TwoTower(qe, de)

    def test_forward_shapes(self):
        """Forward should return two embedding tensors of correct shape."""
        model = self._make_model()
        q = torch.randn(4, 20)
        d = torch.randn(4, 10)
        cats = {"cat1": torch.tensor([0, 1, 2, 0])}
        q_emb, d_emb = model(q, d, cats)
        assert q_emb.shape == (4, 32)
        assert d_emb.shape == (4, 32)

    def test_embeddings_normalized(self):
        """Both embeddings from forward should be L2-normalized."""
        model = self._make_model()
        model.eval()
        q = torch.randn(4, 20)
        d = torch.randn(4, 10)
        cats = {"cat1": torch.tensor([0, 1, 2, 0])}
        q_emb, d_emb = model(q, d, cats)
        q_norms = torch.norm(q_emb, p=2, dim=1)
        d_norms = torch.norm(d_emb, p=2, dim=1)
        assert torch.allclose(q_norms, torch.ones(4), atol=1e-5)
        assert torch.allclose(d_norms, torch.ones(4), atol=1e-5)

    def test_similarity_elementwise(self):
        """Similarity with matching batch dims should give (batch,) output."""
        model = self._make_model()
        model.eval()
        q = torch.randn(4, 20)
        d = torch.randn(4, 10)
        cats = {"cat1": torch.tensor([0, 1, 2, 0])}
        q_emb, d_emb = model(q, d, cats)
        sim = model.similarity(q_emb, d_emb)
        assert sim.shape == (4,)
        # Cosine similarity of unit vectors is in [-1, 1]
        assert (sim >= -1.0 - 1e-6).all()
        assert (sim <= 1.0 + 1e-6).all()

    def test_similarity_pairwise(self):
        """Similarity with different batch dims should give (N, M) output."""
        model = self._make_model()
        model.eval()
        q = torch.randn(3, 20)
        d = torch.randn(5, 10)
        cats_q = {"cat1": torch.tensor([0, 1, 2])}  # not used for query
        cats_d = {"cat1": torch.tensor([0, 1, 2, 0, 1])}
        q_emb = model.query_encoder(q)
        d_emb = model.doc_encoder(d, cats_d)
        sim = model.similarity(q_emb, d_emb)
        assert sim.shape == (3, 5)

    def test_self_similarity(self):
        """A vector's similarity with itself should be ~1.0."""
        model = self._make_model()
        model.eval()
        q = torch.randn(1, 20)
        q_emb = model.query_encoder(q)
        sim = model.similarity(q_emb, q_emb)
        assert torch.allclose(sim, torch.ones(1), atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through both towers."""
        model = self._make_model()
        q = torch.randn(4, 20, requires_grad=True)
        d = torch.randn(4, 10, requires_grad=True)
        cats = {"cat1": torch.tensor([0, 1, 2, 0])}
        q_emb, d_emb = model(q, d, cats)
        loss = (q_emb * d_emb).sum()
        loss.backward()
        assert q.grad is not None
        assert d.grad is not None
        assert q.grad.abs().sum() > 0
        assert d.grad.abs().sum() > 0

    def test_different_emb_dims(self):
        """Should work with various embedding dimensions."""
        for emb_dim in [8, 16, 64, 128]:
            model = self._make_model(emb_dim=emb_dim)
            q = torch.randn(2, 20)
            d = torch.randn(2, 10)
            cats = {"cat1": torch.tensor([0, 1])}
            q_emb, d_emb = model(q, d, cats)
            assert q_emb.shape == (2, emb_dim)
            assert d_emb.shape == (2, emb_dim)
