"""Evaluation workflow for the Two Tower kernel selection model.

Ported from GemmKernelSelection/Embedding/two_tower/workflow/eval.py.
Adapted for tritonBLAS: TwoTowerTrainTask uses separate args
(gemm_features, kernel_continuous, kernel_categoricals, targets).

Functions
---------
evaluate_model     -- evaluate using the training task (in-batch scoring).
evaluate_retrieval -- evaluate using the TwoTowerRetrievalTask (KMeans index).
"""

import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import gmean
from torch import nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from two_tower_triton.data import KSDataset


class FilteredQueryBatchSampler(Sampler):
    """Batch sampler that skips already-processed query IDs (for resume support)."""

    def __init__(self, dataset, processed_qids=None, shuffle=False):
        self.mapper = defaultdict(list)
        for idx in range(len(dataset)):
            qid, _ = dataset.qdary[idx]
            self.mapper[qid].append(idx)

        all_qids = list(self.mapper.keys())
        if processed_qids is not None and len(processed_qids) > 0:
            self.qids = [qid for qid in all_qids if qid not in processed_qids]
            skipped = len(all_qids) - len(self.qids)
            print(f"[FilteredSampler] Skipping {skipped} already processed QIDs")
            print(f"[FilteredSampler] Remaining: {len(self.qids)} QIDs to process")
        else:
            self.qids = all_qids

        self.shuffle = shuffle

    def __iter__(self):
        qids = self.qids.copy()
        if self.shuffle:
            random.shuffle(qids)
        for qid in qids:
            yield self.mapper[qid]

    def __len__(self):
        return len(self.qids)


def evaluate_model(
    model: nn.Module,
    task: nn.Module,
    data_loader: DataLoader,
    dataset: KSDataset,
    device,
    desc: str = "Evaluating",
    save_solutions: bool = False,
    output_path: str = "test_solutions.csv",
    save_every: int = 2000,
) -> dict:
    """Evaluate the model on a dataset and return efficiency metrics.

    For each GEMM problem (query), scores all kernel configs (documents)
    using the training task's forward pass, selects the top-1 prediction,
    and records the true efficiency of that selection.

    Args:
        model: TwoTower model.
        task: TwoTowerTrainTask wrapper.
        data_loader: DataLoader with QueryBatchSampler (one GEMM per batch).
        dataset: KSDataset with raw data for reporting.
        device: Device for inference.
        desc: Progress bar description.
        save_solutions: If True, save per-GEMM predictions to CSV.
        output_path: Path for solutions CSV.
        save_every: Checkpoint interval (number of QIDs).

    Returns:
        dict with keys: eff, mean_eff, gmean_eff, avg_loss, total_evaluated.
    """
    model.eval()
    task.eval()

    processed_qids = set()
    existing_rows = []

    if save_solutions and os.path.isfile(output_path):
        try:
            import pandas as pd
            df_previous = pd.read_csv(output_path)
            processed_qids = set(df_previous["GEMMID"].unique())
            existing_rows = df_previous.to_dict("records")
            print(f"\n{'='*80}")
            print(f"[RESUME MODE] Found existing results at: {output_path}")
            print(f"[RESUME MODE] Already processed: {len(processed_qids)} QIDs")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"[WARNING] Could not load existing results: {e}")

    if save_solutions and len(processed_qids) > 0:
        filtered_sampler = FilteredQueryBatchSampler(
            dataset, processed_qids=processed_qids, shuffle=False
        )
        data_loader = DataLoader(
            dataset,
            batch_sampler=filtered_sampler,
            collate_fn=data_loader.collate_fn,
            num_workers=0,
            pin_memory=True,
        )

    with torch.no_grad():
        pbar = tqdm(data_loader, file=sys.stdout, desc=desc)
        qids = list(data_loader.batch_sampler.qids)

        running_loss = 0.0
        eff = []
        rows = []

        for it, ((x0, x1, x2), y) in enumerate(pbar):
            current_qid = qids[it]

            # Triton TwoTowerTrainTask API: separate args
            loss, logits = task(
                gemm_features=x0.to(device),
                kernel_continuous=x1.to(device),
                kernel_categoricals=None,
                targets=y.to(device),
            )

            predicted_idx = logits.argmax().item()
            efficiency = y[predicted_idx].item()
            eff.append(efficiency)

            running_loss += loss.item()
            avg_loss = running_loss / (it + 1)

            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.5f}",
                    "eff": f"{efficiency:.4f}",
                    "mean_eff": f"{np.mean(eff):.4f}",
                }
            )

            if save_solutions:
                import pandas as pd

                indices = data_loader.batch_sampler.mapper[current_qid]
                pred_dataset_idx = indices[predicted_idx]
                gt_dataset_idx = indices[y.argmax().item()]

                pred_qid, pred_kid = dataset.qdary[pred_dataset_idx]
                gt_qid, gt_kid = dataset.qdary[gt_dataset_idx]

                rows.append(
                    {
                        "GEMMID": current_qid,
                        "PredictedKernelID": int(pred_kid),
                        "GroundTruthKernelID": int(gt_kid),
                        "Efficiency": float(efficiency),
                    }
                )
                if len(rows) >= save_every:
                    _save_checkpoint(rows, existing_rows, processed_qids, output_path)
                    existing_rows.extend(rows)
                    processed_qids.update(r["GEMMID"] for r in rows)
                    rows = []

    if save_solutions and len(rows) > 0:
        _save_checkpoint(rows, existing_rows, processed_qids, output_path)

    eff = np.array(eff).squeeze() if len(eff) > 0 else np.array([0.0])

    stats = {
        "eff": eff,
        "mean_eff": float(eff.mean()),
        "gmean_eff": float(gmean(eff)) if len(eff) > 0 and all(eff > 0) else 0.0,
        "avg_loss": running_loss / len(data_loader) if len(data_loader) > 0 else None,
        "total_evaluated": len(eff),
    }

    if save_solutions:
        _print_summary(stats, output_path, eff)

    return stats


def _save_checkpoint(rows, existing_rows, processed_qids, output_path):
    """Save intermediate results to CSV."""
    import pandas as pd

    df_new = pd.DataFrame(rows)
    if len(existing_rows) > 0:
        df_existing = pd.DataFrame(existing_rows)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(output_path, index=False)
    total_processed = len(processed_qids) + len(rows)
    print(f"\n[Checkpoint] Saved {len(rows)} new solutions, total: {total_processed}")


def _print_summary(stats, output_path, eff):
    """Print evaluation summary."""
    import pandas as pd

    summary_df = pd.DataFrame(
        {
            "mean_eff": [stats["mean_eff"]],
            "gmean_eff": [stats["gmean_eff"]],
            "avg_loss": [stats["avg_loss"]],
            "total_qids": [len(eff)],
        }
    )
    summary_path = output_path.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total QIDs evaluated: {len(eff)}")
    print(f"Mean Efficiency: {stats['mean_eff']:.4f}")
    print(f"Geometric Mean Efficiency: {stats['gmean_eff']:.4f}")
    if stats["avg_loss"]:
        print(f"Average Loss: {stats['avg_loss']:.5f}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")


def evaluate_retrieval(
    retrieval_task,
    dataset: KSDataset,
    device="cpu",
    desc: str = "Retrieval Eval",
) -> dict:
    """Evaluate model using TwoTowerRetrievalTask (KMeans cluster retrieval).

    For each GEMM in the dataset, uses the retrieval task's brute-force
    retrieve_all to rank all kernel configs, selects top-1, and looks up
    its true efficiency.

    Args:
        retrieval_task: TwoTowerRetrievalTask instance (pre-built index).
        dataset: KSDataset to evaluate on.
        device: Device for inference.
        desc: Progress bar description.

    Returns:
        dict with keys: eff, mean_eff, gmean_eff, top1_accuracy,
        selection_efficiency, total_evaluated.
    """
    from two_tower_triton.data import GEMMDataset
    from torch.utils.data import DataLoader

    gemm_dataset = GEMMDataset(dataset)
    gemm_loader = DataLoader(gemm_dataset, batch_size=1, shuffle=False)

    eff_list = []
    best_eff_list = []

    pbar = tqdm(gemm_loader, file=sys.stdout, desc=desc)
    for qid_tensor, query_features in pbar:
        qid = qid_tensor.item() if hasattr(qid_tensor, "item") else int(qid_tensor[0])
        q_feat = query_features.squeeze(0).numpy()

        # Retrieve all kernel configs ranked by similarity
        sorted_indices, scores = retrieval_task.retrieve_all(q_feat)

        # Get the mapping from dataset kernel indices to KernelIDs
        # The retrieval task uses doc_embeddings indexed 0..N-1
        # We need to map back to the dataset's efficiency data
        if qid in dataset.query_map:
            # Find all (qid, kid) pairs in the dataset
            qid_mask = dataset.qdary[:, 0] == qid
            qid_rows = dataset.qdary[qid_mask]
            if len(qid_rows) > 0:
                effs = dataset.y.iloc[np.where(qid_mask)[0]]["EFF"].values

                # Best possible efficiency for this GEMM
                best_eff = float(effs.max())
                best_eff_list.append(best_eff)

                # Selected efficiency: top-1 from retrieval
                # sorted_indices[0] is the best doc index
                # Map it back to the dataset if possible
                if sorted_indices[0] < len(effs):
                    selected_eff = float(effs[sorted_indices[0]])
                else:
                    selected_eff = float(effs[0])  # fallback
                eff_list.append(selected_eff)

                pbar.set_postfix(
                    {
                        "eff": f"{selected_eff:.4f}",
                        "mean_eff": f"{np.mean(eff_list):.4f}",
                    }
                )

    eff = np.array(eff_list) if len(eff_list) > 0 else np.array([0.0])
    best_effs = np.array(best_eff_list) if len(best_eff_list) > 0 else np.array([1.0])

    # top1_accuracy: fraction where selected config achieves >= 0.99 of best
    top1_acc = float((eff >= 0.99 * best_effs).mean()) if len(eff) > 0 else 0.0

    # selection_efficiency: mean ratio of selected / best
    sel_eff = float((eff / np.maximum(best_effs, 1e-8)).mean()) if len(eff) > 0 else 0.0

    stats = {
        "eff": eff,
        "mean_eff": float(eff.mean()),
        "gmean_eff": float(gmean(eff)) if len(eff) > 0 and all(eff > 0) else 0.0,
        "top1_accuracy": top1_acc,
        "selection_efficiency": sel_eff,
        "total_evaluated": len(eff),
    }

    return stats
