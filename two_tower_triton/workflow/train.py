"""Training workflow for the Two Tower kernel selection model.

Ported from GemmKernelSelection/Embedding/two_tower/workflow/train.py.
Adapted for tritonBLAS: TwoTowerTrainTask uses separate args
(gemm_features, kernel_continuous, kernel_categoricals, targets) instead
of the tuple ((x0, x1, x2), y) from the reference.

Functions
---------
train_model -- full training loop with validation and early stopping.
"""

import sys
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from two_tower_triton.data import KSDataset
from .eval import evaluate_model


def train_model(
    model: nn.Module,
    task: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_dataset: KSDataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    n_epochs: int,
    patience: int,
    min_delta: float,
    best_model_path: str,
) -> Tuple[float, int]:
    """Train the Two Tower model with validation and early stopping.

    Args:
        model: The TwoTower model.
        task: TwoTowerTrainTask wrapper (loss + auxiliary head).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data (with QueryBatchSampler).
        val_dataset: Validation KSDataset instance.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        device: Device to run training on.
        n_epochs: Maximum number of epochs.
        patience: Early stopping patience (epochs without improvement).
        min_delta: Minimum improvement for early stopping.
        best_model_path: Path to save the best model checkpoint.

    Returns:
        Tuple of (best_gmean_eff, best_epoch).
    """
    best_gmean = 0.0
    best_epoch = 0
    early_stop_counter = 0
    best_train_loss = 1e3

    for epoch in range(n_epochs):
        pbar = tqdm(train_loader, file=sys.stdout)
        running_loss = 0.0
        model.train()
        task.train()

        for it, ((x0, x1, x2), y) in enumerate(pbar):
            optimizer.zero_grad()

            # Triton TwoTowerTrainTask API: separate args
            loss, _ = task(
                gemm_features=x0.to(device),
                kernel_continuous=x1.to(device),
                kernel_categoricals=None,  # x2 is edoc (int32), unused for now
                targets=y.to(device),
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{(running_loss / (it + 1)):.5f}"})

        scheduler.step()

        # If no validation queries, use train loss for early stopping
        if len(val_dataset.queries) == 0:
            print("No validation queries available, skipping validation.")
            train_loss = running_loss / (it + 1)

            if best_train_loss - train_loss > min_delta:
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"Epoch {epoch:02d} ==> New best model saved with Loss: {train_loss:.6f}"
                )
                best_train_loss = train_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter} epochs")

            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            continue

        # Validation
        val_stats = evaluate_model(
            model, task, val_loader, val_dataset, device, "Validation"
        )
        val_gmean = val_stats["gmean_eff"]
        print(
            f"Epoch {epoch:02d} ==> Mean VAL EFF ==> {val_stats['mean_eff']:.4f} "
            f"| Geo-Mean VAL EFF ==> {val_gmean:.4f}"
        )

        # Early stopping check (higher gmean = better)
        if val_gmean > best_gmean + min_delta:
            best_gmean = val_gmean
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Geo-Mean EFF: {best_gmean:.4f}")
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epochs")

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return best_gmean, best_epoch
