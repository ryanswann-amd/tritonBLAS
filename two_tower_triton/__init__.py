"""Two Tower embedding model for tritonBLAS kernel selection.

This package ports the Two Tower architecture from GemmKernelSelection
(ROCm/GemmKernelSelection custom_features branch) and adapts it for
Triton kernel configuration selection in tritonBLAS.

Architecture:
  - Query tower: encodes GEMM problem shapes (M, N, K, batch, dtype, layout)
    into a fixed-size embedding using engineered features (log dims, arithmetic
    intensity, roofline estimates, cache pressure, wave alignment, etc.)
  - Document tower: encodes Triton kernel configurations (tile sizes, warps,
    stages, split-K, etc.) into the same embedding space
  - Similarity: dot product between query and document embeddings predicts
    kernel efficiency for the given GEMM problem

Modules:
  config  - YAML-based configuration (Config class)
  model   - Neural network components (MLP, TwoTower)
  task    - Training and retrieval tasks
  workflow - Training and evaluation pipelines
"""

from .config.config import Config, print_config
from .model.mlp import MLP

__all__ = ["Config", "print_config", "MLP"]
