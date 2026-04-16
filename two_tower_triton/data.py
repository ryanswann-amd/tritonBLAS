"""Data loading and dataset classes for Two Tower kernel selection model.

Ported from GemmKernelSelection/Embedding/two_tower/data.py, adapted for
tritonBLAS Triton config space.

Classes
-------
KSData   -- loads gemms.csv, kernels.csv, perf.csv; applies feature engineering,
           normalizes, and splits into train/val/test.
KSDataset -- PyTorch Dataset that yields (query, doc, edoc), eff tuples.
QueryBatchSampler -- groups samples by GEMM id for in-batch negatives.
GEMMDataset -- iterates over unique GEMM problems (for inference/eval).
"""

import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler

from two_tower_triton.features import (
    gemm_preprocess,
    kernel_preprocess,
    get_continuous_gemm_cols,
    get_continuous_kernel_cols,
    get_categorical_kernel_cols,
)

# -- Triton kernel config columns ------------------------------------------
# Raw columns expected in kernels.csv for the Triton config space.
TRITON_KERNEL_COLS = [
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
    "waves_per_eu",
    "matrix_instr_nonkdim",
]

# Raw GEMM problem columns expected in gemms.csv.
GEMM_COLS = ["M", "N", "K"]

# Optional extra GEMM columns that may be present.
GEMM_OPT_COLS = ["batch_count", "lda", "stride_a", "ldb", "stride_b", "ldc", "stride_c"]


# -- Helpers ----------------------------------------------------------------

def get_mean_std(df, indices=None):
    """Compute per-column mean and std; replace zero std with 1."""
    if indices is not None:
        df = df.loc[indices]
    mean = df.mean()
    std = df.std()
    std[std == 0] = 1
    return mean, std


def min_assignment(universe, subsets):
    """Greedy set-cover: return minimal list of subset IDs covering *universe*.

    Parameters
    ----------
    universe : set
        Set of element IDs that must be covered.
    subsets : list[tuple[id, set]]
        Each element is (subset_id, set_of_elements_covered).

    Returns
    -------
    list | None
        Subset IDs selected, or None if coverage is impossible.
    """
    elements = set(e for s in subsets for e in s[1])
    if not universe.issubset(elements):
        return None
    covered = set()
    cover = []
    while covered != universe:
        subset = max(subsets, key=lambda s: len(s[1] - covered))
        if len(subset[1] - covered) == 0:
            break
        cover.append(subset[0])
        covered |= subset[1]
    return cover


# -- KSData -----------------------------------------------------------------

class KSData:
    """Load and preprocess GEMM + kernel + performance data for Two Tower training.

    Expects a data directory containing:
      - gemms.csv   : GEMM problem descriptors (GEMMID, M, N, K[, batch_count, ...])
      - kernels.csv : Triton kernel configs (KernelID, BLOCK_SIZE_M, ...)
      - perf.csv    : Performance mapping (GEMMID, KernelID, EFF[, us])

    Parameters
    ----------
    data_dir : str
        Path to directory with the three CSV files.
    gpu_arch : str
        GPU architecture key (``"mi300x"`` or ``"mi350x"``).
    dtype : str
        Data type string (e.g. ``"bf16"``, ``"fp16"``).
    split : tuple[float, float, float]
        Train/val/test proportions.  Must sum to 1.
    seed : int
        Random seed for reproducibility.
    kernel_min_eff : float | None
        If set, apply greedy set-cover to keep only kernels that are
        ``>= kernel_min_eff`` efficient on at least one problem. This
        dramatically reduces the kernel space.
    drop_low_eff : bool
        Whether to drop training rows with EFF below *drop_low_eff_thr*.
    drop_low_eff_thr : float
        Threshold for dropping low-efficiency rows.
    """

    def __init__(
        self,
        data_dir,
        gpu_arch="mi300x",
        dtype="bf16",
        split=(0.8, 0.1, 0.1),
        seed=0,
        kernel_min_eff=0.99,
        drop_low_eff=False,
        drop_low_eff_thr=0.3,
    ):
        self._validate_split(split)
        self.gpu_arch = gpu_arch
        self.dtype = dtype

        # --- Load raw data ---
        gdf = self._load_csv(os.path.join(data_dir, "gemms.csv"), index_col="GEMMID")
        kdf = self._load_csv(os.path.join(data_dir, "kernels.csv"), index_col="KernelID")
        pdf = self._load_csv(
            os.path.join(data_dir, "perf.csv"), index_col=["GEMMID", "KernelID"]
        ).astype(np.float32)

        # Keep only GEMMs present in perf data
        avail = pdf.index.get_level_values("GEMMID").unique()
        gdf = gdf.loc[gdf.index.intersection(avail)]

        self.gdf_raw = gdf.copy()
        self.kdf_raw = kdf.copy()

        # --- Feature engineering ---
        # Rename columns to match features module expectations (M, N, K)
        gcols_rename = {}
        for c in gdf.columns:
            cl = c.lower()
            if cl == "m":
                gcols_rename[c] = "M"
            elif cl == "n":
                gcols_rename[c] = "N"
            elif cl == "k":
                gcols_rename[c] = "K"
        if gcols_rename:
            gdf = gdf.rename(columns=gcols_rename)

        gdf = gemm_preprocess(gdf, gpu_arch=gpu_arch, dtype=dtype)
        kdf = kernel_preprocess(kdf, dtype=dtype)

        # Ensure float32
        gdf = gdf.astype(np.float32)
        kdf = kdf.astype(np.float32)

        print(f"[INFO] # GEMM FEATURES   ==> {len(gdf.columns)}")
        print(f"[INFO] # KERNEL FEATURES ==> {len(kdf.columns)}")

        # Keep only kernels present in perf
        perf_kid = pdf.index.get_level_values("KernelID").unique()
        kdf = kdf.loc[kdf.index.intersection(perf_kid)]

        # --- Kernel minimization (greedy set-cover) ---
        if kernel_min_eff is not None:
            pdf, kdf = self._apply_kernel_minimization(pdf, kdf, kernel_min_eff)

        self.gdf = gdf
        self.kdf = kdf
        self.eff = pd.DataFrame(pdf[["EFF"]])
        if "us" in pdf.columns:
            self.eff["us"] = pdf["us"]

        # --- Train/val/test split (by GEMM, not row) ---
        self.train_indices, self.val_indices, self.test_indices = self._split_indices(
            split, seed
        )

        # --- Normalize continuous features ---
        train_kid = self.eff.loc[self.train_indices].index.get_level_values(
            "KernelID"
        ).unique()
        self.gmu, self.gstd = get_mean_std(self.gdf, self.train_indices)
        self.kmu, self.kstd = get_mean_std(self.kdf, train_kid)

        for dset in ("train", "val", "test"):
            indices = getattr(self, f"{dset}_indices")
            queries = (self.gdf.loc[indices] - self.gmu) / self.gstd
            y = self.eff.loc[indices]

            if drop_low_eff and dset == "train":
                mask = y["EFF"] >= drop_low_eff_thr
                dropped = (~mask).sum()
                print(
                    f"[INFO] Dropping {dropped} rows with EFF < {drop_low_eff_thr} from train"
                )
                y = y[mask]

            kid_in_split = y.index.get_level_values("KernelID").unique()
            docs = (self.kdf.loc[self.kdf.index.intersection(kid_in_split)] - self.kmu) / self.kstd

            setattr(
                self,
                dset,
                dict(
                    queries=queries,
                    docs=docs,
                    edocs=None,
                    y=y,
                    gdf=self.gdf_raw,
                    kdf=self.kdf_raw,
                ),
            )

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _validate_split(split):
        if not isinstance(split, (tuple, list)) or len(split) != 3:
            raise ValueError("split must be a 3-element tuple/list (train, val, test)")
        if not np.isclose(sum(split), 1.0):
            raise ValueError("split proportions must sum to 1")

    @staticmethod
    def _load_csv(path, index_col=None):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.read_csv(path, index_col=index_col)

    def _split_indices(self, split, seed):
        """Split GEMM IDs into train/val/test -- same GEMM never appears in two sets."""
        ids = self.gdf.index
        train_ids, remainder = train_test_split(
            ids, test_size=1 - split[0], shuffle=True, random_state=seed
        )
        if split[2] == 0:
            return train_ids, remainder, pd.Index([])
        val_size = split[1] / (split[1] + split[2])
        val_ids, test_ids = train_test_split(
            remainder, test_size=1 - val_size, shuffle=True, random_state=seed
        )
        return train_ids, val_ids, test_ids

    @staticmethod
    def _apply_kernel_minimization(pdf, kdf, threshold):
        """Keep only the minimal set of kernels covering all GEMMs at >= threshold EFF."""
        dff = pdf[pdf["EFF"] >= threshold]
        if dff.empty:
            print(
                f"[WARN] No kernel achieves EFF >= {threshold}; skipping minimization."
            )
            return pdf, kdf

        problems = set(dff.index.get_level_values("GEMMID").unique())
        subsets = []
        for kid, rows in dff.groupby(level="KernelID"):
            covered = set(rows.index.get_level_values("GEMMID").values)
            subsets.append((kid, covered))

        selected = min_assignment(problems, subsets)
        if selected is None:
            print("[WARN] min_assignment could not cover all problems; keeping all kernels.")
            return pdf, kdf

        mask = pdf.index.get_level_values("KernelID").isin(selected)
        pdf = pdf[mask].copy()
        kdf = kdf.loc[kdf.index.intersection(
            pdf.index.get_level_values("KernelID").unique()
        )]
        print(
            f"[INFO] Kernel minimization (EFF >= {threshold}): {len(kdf)} kernels selected"
        )
        return pdf, kdf


# -- KSDataset --------------------------------------------------------------

class KSDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed Two Tower data.

    Each sample is a (GEMM, kernel) pair with an efficiency label.

    Returns
    -------
    (query, doc, edoc), eff
        - query : np.ndarray[float32]   -- normalised GEMM features
        - doc   : np.ndarray[float32]   -- normalised kernel features
        - edoc  : np.ndarray[int32]     -- categorical kernel features (for nn.Embedding)
        - eff   : float32               -- efficiency label
    """

    def __init__(self, data, emb_cols=None, train=True):
        super().__init__()
        self.queries = data["queries"]
        self.docs = data["docs"]
        self.edocs = data["edocs"]
        self.y = data["y"]
        self.gdf = data["gdf"]
        self.kdf = data["kdf"]
        self.train = train
        self.emb_cols = emb_cols or {}

        # Pre-materialise numpy arrays for fast lookup
        self.query_map = {idx: row.values for idx, row in self.queries.iterrows()}
        self.doc_map = {idx: row.values for idx, row in self.docs.iterrows()}
        self.edoc_map = (
            {idx: row.values for idx, row in self.edocs.iterrows()}
            if self.edocs is not None
            else {}
        )

        # (GEMMID, KernelID) pairs
        self.qdary = np.array([idx for idx in self.y.index.values])

        # Map kernel IDs -> sequential indices for nn.Embedding
        self.kernel_id_to_idx = {kid: i for i, kid in enumerate(self.kdf.index)}

    def input_sizes(self):
        """Return (query_dim, (doc_continuous_dim, emb_cols_dict))."""
        return self.queries.shape[1], (self.docs.shape[1], self.emb_cols)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        qid, did = self.qdary[index]
        score = self.y.iloc[index]["EFF"]
        query = self.query_map[qid]
        doc = self.doc_map[did]
        edoc = self.edoc_map.get(did, np.empty(0)).astype(np.int32)
        return (query, doc, edoc), score


# -- QueryBatchSampler ------------------------------------------------------

class QueryBatchSampler(Sampler):
    """Groups samples by GEMM ID so each batch contains all kernels for one problem.

    This enables efficient in-batch negative sampling during contrastive training.
    """

    def __init__(self, dataset, shuffle=True):
        self.mapper = defaultdict(list)
        for idx in range(len(dataset)):
            qid, _ = dataset.qdary[idx]
            self.mapper[qid].append(idx)
        self.qids = list(self.mapper.keys())
        self.shuffle = shuffle

    def __iter__(self):
        qids = self.qids.copy()
        if self.shuffle:
            random.shuffle(qids)
        for qid in qids:
            yield self.mapper[qid]

    def __len__(self):
        return len(self.qids)


# -- GEMMDataset ------------------------------------------------------------

class GEMMDataset(Dataset):
    """Iterates over unique GEMM problems (for inference / evaluation)."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.query_ids = list(dataset.query_map.keys())

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, index):
        qid = self.query_ids[index]
        return qid, self.dataset.query_map[qid]
