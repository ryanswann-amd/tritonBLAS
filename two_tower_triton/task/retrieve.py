"""Retrieval / inference task for the Two Tower kernel selection model.

Ported from GemmKernelSelection/Embedding/two_tower/task/retrieve.py.
Adapted for tritonBLAS: uses the adapted TwoTower API with separate
query_features, doc_continuous, and doc_categoricals arguments.

TwoTowerRetrievalTask: Pre-computes kernel embeddings, builds KMeans index,
    then retrieves top-K configs for a given GEMM query via cluster search.
"""

import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans

from ..model.model import TwoTower


def get_doc_embeddings(model, kernel_continuous, kernel_categoricals=None, device="cpu"):
    """Compute document embeddings for all kernel configs.

    Args:
        model: TwoTower model instance.
        kernel_continuous: (N, continuous_dim) numpy array or tensor of
            continuous kernel config features.
        kernel_categoricals: Dict mapping column name to (N,) numpy array
            or tensor of category indices. Can be None.
        device: Device to run inference on.

    Returns:
        embeddings: (N, emb_dim) numpy array of L2-normalized embeddings.
    """
    model.eval()
    with torch.no_grad():
        doc_feat = torch.as_tensor(kernel_continuous, dtype=torch.float32, device=device)

        cat_tensors = None
        if kernel_categoricals is not None and len(kernel_categoricals) > 0:
            cat_tensors = {
                name: torch.as_tensor(vals, dtype=torch.long, device=device)
                for name, vals in kernel_categoricals.items()
            }

        model.to(device)
        embedding = model.doc_encoder(doc_feat, cat_tensors).cpu().numpy()
    return embedding


class TwoTowerRetrievalTask(nn.Module):
    """Inference for the Two Tower model with KMeans cluster index.

    Pre-computes all kernel config embeddings at init time, clusters them
    with KMeans, and at query time searches the nearest cluster(s) to
    find the best matching kernel config.

    Args:
        model: Trained TwoTower model.
        kernel_configs: DataFrame or dict of kernel configs (for returning
            human-readable results).
        kernel_continuous: (N, continuous_dim) numpy array of kernel features.
        kernel_categoricals: Dict of categorical arrays, or None.
        scaler: Dict with 'query_mean', 'query_std', 'doc_mean', 'doc_std'
            for normalizing features.
        n_clusters: Number of KMeans clusters for the embedding index.
        device: Device for inference.
    """

    def __init__(
        self,
        model: TwoTower,
        kernel_configs,
        kernel_continuous: np.ndarray,
        kernel_categoricals: dict = None,
        scaler: dict = None,
        n_clusters: int = 10,
        device: str = "cpu",
    ):
        super().__init__()
        self.model = model
        self.kernel_configs = kernel_configs
        self.scaler = scaler or {}
        self.device = device
        self.n_clusters = n_clusters

        # Pre-compute document embeddings for all kernel configs
        self.doc_embeddings = get_doc_embeddings(
            model, kernel_continuous, kernel_categoricals, device=device
        )

        # Build KMeans index over document embeddings
        n_actual_clusters = min(n_clusters, len(self.doc_embeddings))
        self.kmeans = KMeans(
            n_clusters=n_actual_clusters, random_state=42, n_init=10
        )
        cluster_labels = self.kmeans.fit_predict(self.doc_embeddings)

        # Build posting lists: cluster_id -> list of doc indices
        self.posting_lists = {i: [] for i in range(n_actual_clusters)}
        for idx, label in enumerate(cluster_labels):
            self.posting_lists[label].append(idx)

    def _encode_query(self, query_features: np.ndarray) -> np.ndarray:
        """Encode query features through the query tower.

        Args:
            query_features: (1, query_dim) or (query_dim,) numpy array.

        Returns:
            (1, emb_dim) numpy array of L2-normalized query embedding.
        """
        self.model.eval()
        with torch.no_grad():
            if query_features.ndim == 1:
                query_features = query_features[np.newaxis, :]
            q_tensor = torch.as_tensor(
                query_features, dtype=torch.float32, device=self.device
            )
            self.model.to(self.device)
            q_emb = self.model.query_encoder(q_tensor).cpu().numpy()
        return q_emb

    def select_config(self, query_features: np.ndarray, top_k: int = 1):
        """Select best kernel config(s) for a given GEMM problem.

        Finds the nearest KMeans centroid, then searches within that cluster
        for the highest dot-product similarity kernel config. Falls back to
        the next cluster if the best cluster is empty.

        Args:
            query_features: (query_dim,) or (1, query_dim) preprocessed
                and normalized GEMM features.
            top_k: Number of configs to return.

        Returns:
            List of dicts, each with:
                - 'index': kernel index in the original config array
                - 'score': dot-product similarity score
                - 'config': config dict (if kernel_configs is a DataFrame)
        """
        q_emb = self._encode_query(query_features)  # (1, emb_dim)

        # Find nearest centroids
        centroid_sims = [
            (i, (q_emb[0] * centroid).sum())
            for i, centroid in enumerate(self.kmeans.cluster_centers_)
        ]
        sorted_centroids = sorted(centroid_sims, key=lambda x: x[1], reverse=True)

        # Collect top-k results across clusters
        results = []
        for centroid_idx, _ in sorted_centroids:
            cluster_docs = self.posting_lists[centroid_idx]
            if not cluster_docs:
                continue

            # Compute similarities within this cluster
            cluster_embs = self.doc_embeddings[cluster_docs]  # (C, emb_dim)
            sims = (q_emb @ cluster_embs.T).squeeze()  # (C,)
            if sims.ndim == 0:
                sims = np.array([sims.item()])

            # Sort by similarity descending
            sorted_idx = np.argsort(-sims)
            for si in sorted_idx:
                doc_idx = cluster_docs[si]
                score = float(sims[si])

                result = {"index": doc_idx, "score": score}
                # Attach config details if available
                if hasattr(self.kernel_configs, "iloc"):
                    result["config"] = self.kernel_configs.iloc[doc_idx].to_dict()
                elif isinstance(self.kernel_configs, dict):
                    result["config"] = {
                        k: v[doc_idx] if hasattr(v, "__getitem__") else v
                        for k, v in self.kernel_configs.items()
                    }

                results.append(result)
                if len(results) >= top_k:
                    return results

        return results

    def retrieve_all(self, query_features: np.ndarray):
        """Brute-force retrieval: rank ALL kernel configs by similarity.

        Useful for evaluation (computing geo-mean efficiency over all shapes).

        Args:
            query_features: (query_dim,) or (1, query_dim) preprocessed features.

        Returns:
            Tuple of (indices, scores):
                indices: (N,) numpy array of kernel indices sorted by similarity desc.
                scores: (N,) numpy array of corresponding similarity scores.
        """
        q_emb = self._encode_query(query_features)  # (1, emb_dim)
        sims = (q_emb @ self.doc_embeddings.T).squeeze()  # (N,)
        sorted_idx = np.argsort(-sims)
        return sorted_idx, sims[sorted_idx]

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu"):
        """Load a TwoTowerRetrievalTask from a saved checkpoint.

        Checkpoint must contain:
            - model_state_dict: state dict for the TwoTower model
            - config: dict with model architecture params
            - kernel_configs: DataFrame.to_dict() or dict of kernel configs
            - kernel_features: numpy array of preprocessed kernel features
            - scaler: normalization parameters
            - kernel_categoricals (optional): dict of categorical arrays

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            device: Device for inference.

        Returns:
            TwoTowerRetrievalTask instance ready for inference.
        """
        import pandas as pd

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        config = ckpt["config"]

        # Reconstruct model from config
        from ..model.model import QueryEncoder, DocumentEncoder

        query_encoder = QueryEncoder(
            in_dim=config.get("query_in_dim", 50),
            hidden_sizes=config.get("query_hidden", [256, 384, 256]),
            emb_dim=config.get("emb_dim", 128),
        )
        doc_encoder = DocumentEncoder(
            continuous_dim=config.get("doc_continuous_dim", 20),
            cat_dims=config.get("doc_cat_dims", {}),
            emb_dim=config.get("emb_dim", 128),
            hidden_sizes=config.get("doc_hidden", [128, 256, 128]),
        )
        model = TwoTower(query_encoder, doc_encoder)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Restore kernel configs
        kernel_configs_raw = ckpt.get("kernel_configs", {})
        if isinstance(kernel_configs_raw, dict):
            kernel_configs = pd.DataFrame(kernel_configs_raw)
        else:
            kernel_configs = kernel_configs_raw

        kernel_features = ckpt.get("kernel_features", np.zeros((0, 0)))
        if isinstance(kernel_features, torch.Tensor):
            kernel_features = kernel_features.numpy()

        scaler = ckpt.get("scaler", {})
        kernel_categoricals = ckpt.get("kernel_categoricals", None)

        return cls(
            model=model,
            kernel_configs=kernel_configs,
            kernel_continuous=kernel_features,
            kernel_categoricals=kernel_categoricals,
            scaler=scaler,
            device=device,
        )


def save_checkpoint(model, scaler, kernel_configs, kernel_features, config, path,
                    kernel_categoricals=None):
    """Save everything needed for inference to a single .pt file.

    Args:
        model: Trained TwoTower model.
        scaler: Dict with normalization stats.
        kernel_configs: DataFrame or dict of kernel configurations.
        kernel_features: numpy array of preprocessed kernel features.
        config: Dict of model architecture config.
        path: Output file path.
        kernel_categoricals: Optional dict of categorical arrays.
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "kernel_configs": (
            kernel_configs.to_dict()
            if hasattr(kernel_configs, "to_dict")
            else kernel_configs
        ),
        "kernel_features": kernel_features,
        "config": config,
        "version": "1.0",
    }
    if kernel_categoricals is not None:
        save_dict["kernel_categoricals"] = kernel_categoricals

    torch.save(save_dict, path)
