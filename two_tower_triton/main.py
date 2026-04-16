"""Main entry point for Two Tower tritonBLAS kernel selection model.

Adapted from GemmKernelSelection/Embedding/main.py for the Triton
kernel config space.

Usage
-----
    python -m two_tower_triton.main --config two_tower_triton/config/config.yaml \
        --data_dir /path/to/data --mode train_eval --output_dir output/

Modes
-----
    train      : Train the model, save best checkpoint.
    eval       : Load checkpoint, evaluate on test set.
    train_eval : Train then evaluate.
    retrieve   : Build TwoTowerRetrievalTask, evaluate retrieval quality.

Output metrics (parseable by research_loop experiment_command):
    top1_accuracy: 0.XX
    mean_eff: 0.XX
    selection_efficiency: 0.XX
    val_loss: 0.XX
"""

import argparse
import gc
import os
import random

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import gmean
from torch.utils.data import DataLoader

from two_tower_triton.config.config import Config, print_config
from two_tower_triton.data import KSData, KSDataset, QueryBatchSampler
from two_tower_triton.model.model import TwoTower, QueryEncoder, DocumentEncoder
from two_tower_triton.task.task import TwoTowerTrainTask
from two_tower_triton.task.retrieve import TwoTowerRetrievalTask
from two_tower_triton.workflow.train import train_model
from two_tower_triton.workflow.eval import evaluate_model, evaluate_retrieval


def set_seed(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")


def build_model(input_sizes, config):
    """Construct TwoTower model from config and data-derived input sizes.

    Args:
        input_sizes: Tuple (query_dim, (doc_continuous_dim, emb_cols_dict))
            returned by KSDataset.input_sizes().
        config: Config object.

    Returns:
        TwoTower model instance.
    """
    query_dim = input_sizes[0]
    doc_continuous_dim, emb_cols = input_sizes[1]

    # Convert emb_cols to cat_dims dict for DocumentEncoder
    cat_dims = {}
    if emb_cols:
        if isinstance(emb_cols, dict):
            cat_dims = emb_cols
        elif hasattr(emb_cols, "to_dict"):
            cat_dims = emb_cols.to_dict()

    query_encoder = QueryEncoder(
        in_dim=query_dim,
        hidden_sizes=config.model.layer_sizes_query,
        emb_dim=config.model.emb_dim,
        dropout_rate=config.model.dropout_rate,
        use_norm=config.model.norm,
    )

    doc_encoder = DocumentEncoder(
        continuous_dim=doc_continuous_dim,
        cat_dims=cat_dims,
        emb_dim=config.model.emb_dim,
        hidden_sizes=config.model.layer_sizes_doc,
        dropout_rate=config.model.dropout_rate,
        use_norm=config.model.norm,
    )

    return TwoTower(query_encoder, doc_encoder)


def generate_best_model_path(output_path, dtype, mo, suffix):
    """Generate path for saving the best model checkpoint."""
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, f"{dtype}_{mo}_{suffix}.pth")


def main(config, mode="train_eval", checkpoint=None, output_dir=None):
    """Main training/evaluation pipeline.

    Args:
        config: Config object loaded from YAML.
        mode: One of 'train', 'eval', 'train_eval', 'retrieve'.
        checkpoint: Path to model checkpoint (for eval/retrieve modes).
        output_dir: Override output directory.
    """
    seed = config.data.seed if hasattr(config.data, "seed") else 42
    set_seed(seed)

    # Resolve paths
    data_path = config.io.data_path
    out_path = output_dir or config.io.output_path
    if out_path:
        os.makedirs(out_path, exist_ok=True)

    dtype = config.data.dtype
    mo = config.data.get("mo", "TN") if hasattr(config.data, "get") else "TN"
    gpu_arch = config.data.gpu_arch

    # Device
    device_str = config.training.device if "training" in config else "cpu"
    if device_str != "cpu" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] GPU arch: {gpu_arch}")

    # Determine if we need training or eval configs
    do_train = mode in ("train", "train_eval")
    do_eval = mode in ("eval", "train_eval")
    do_retrieve = mode == "retrieve"

    # Validate
    if not data_path or not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data directory not found: {data_path}. "
            "Set io.data_path in config or use --data_dir."
        )

    if (do_eval or do_retrieve) and not do_train and not checkpoint:
        raise ValueError(
            "Evaluation/retrieval without training requires --checkpoint."
        )

    # --- Load data ---
    emb_cols = {}
    if "emb_cols" in config.data:
        ec = config.data.emb_cols
        if hasattr(ec, "to_dict"):
            emb_cols = ec.to_dict()
        elif isinstance(ec, dict):
            emb_cols = ec

    kernel_min_eff = config.data.get("kernel_minimization_thr", None)
    split = config.data.split if hasattr(config.data, "split") else [0.7, 0.15, 0.15]

    data = KSData(
        data_path,
        gpu_arch=gpu_arch,
        dtype=dtype,
        split=tuple(split),
        seed=seed,
        kernel_min_eff=kernel_min_eff,
    )

    # --- Build datasets ---
    if do_train:
        train_dataset = KSDataset(data.train, emb_cols=emb_cols, train=True)
        val_dataset = KSDataset(data.val, emb_cols=emb_cols, train=False)
        val_sampler = QueryBatchSampler(val_dataset, shuffle=False)

    if do_eval or do_retrieve:
        test_dataset = KSDataset(data.test, emb_cols=emb_cols, train=False)
        test_sampler = QueryBatchSampler(test_dataset, shuffle=False)

    # --- Build model ---
    ref_dataset = train_dataset if do_train else test_dataset
    input_sizes = ref_dataset.input_sizes()
    model = build_model(input_sizes, config).to(device)
    task = TwoTowerTrainTask(model).to(device)

    # --- Training config ---
    n_epochs = config.training.n_epochs if "training" in config else 100
    patience = config.training.early_stop.patience if "training" in config else 15
    min_delta = config.training.early_stop.min_delta if "training" in config else 1e-4
    batch_size = config.training.batch_size if "training" in config else 256
    num_workers = config.training.get("num_workers", 4) if "training" in config else 4

    best_model_path = generate_best_model_path(
        out_path, dtype, mo, config.io.get("best_model_suffix", "model")
    )

    best_gmean = 0.0
    best_epoch = 0

    # --- Train ---
    if do_train:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Optimizer
        lr = config.optimizer.params.lr if "optimizer" in config else 1e-3
        weight_decay = config.optimizer.params.get("weight_decay", 1e-4) if "optimizer" in config else 1e-4
        optimizer = optim.Adam(task.parameters(), lr=lr, weight_decay=weight_decay)

        # OneCycleLR scheduler
        steps_per_epoch = len(train_loader)
        max_lr = lr * 3
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        print(
            f"[INFO] OneCycleLR: max_lr={max_lr:.2e}, "
            f"{n_epochs} epochs, {steps_per_epoch} steps/epoch"
        )

        best_gmean, best_epoch = train_model(
            model=model,
            task=task,
            train_loader=train_loader,
            val_loader=val_loader,
            val_dataset=val_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            n_epochs=n_epochs,
            patience=patience,
            min_delta=min_delta,
            best_model_path=best_model_path,
        )
        print(f"\n[INFO] Training complete. Best Geo-Mean EFF: {best_gmean:.4f} at epoch {best_epoch}")

    # --- Evaluate ---
    if do_eval or do_retrieve:
        # Load best model
        load_path = checkpoint if (not do_train and checkpoint) else best_model_path
        if os.path.isfile(load_path):
            print(f"[INFO] Loading model: {load_path}")
            model.load_state_dict(torch.load(load_path, weights_only=True, map_location=device))
            model.to(device)
        else:
            print(f"[WARN] Model file not found: {load_path}, using current weights")

        if do_eval and len(test_dataset) > 0:
            print("\n[INFO] Final Evaluation on Test Dataset")
            torch.cuda.empty_cache()
            gc.collect()

            test_loader = DataLoader(
                test_dataset,
                batch_sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_stats = evaluate_model(
                model,
                task,
                test_loader,
                test_dataset,
                device,
                "Testing",
                save_solutions=True,
                output_path=os.path.join(out_path, "test_solutions.csv"),
            )

            eff = test_stats["eff"]
            # Parseable metric lines for research_loop
            print(f"\ntop1_accuracy: {(eff >= 0.99).mean():.4f}")
            print(f"mean_eff: {test_stats['mean_eff']:.4f}")
            print(f"selection_efficiency: {test_stats['gmean_eff']:.4f}")
            if test_stats["avg_loss"] is not None:
                print(f"val_loss: {test_stats['avg_loss']:.4f}")
            print(f"gmean_eff: {test_stats['gmean_eff']:.4f}")

        if do_retrieve:
            print("\n[INFO] Retrieval Evaluation")
            retrieval_task = TwoTowerRetrievalTask(
                model=model,
                kernel_configs=data.kdf_raw,
                kernel_continuous=(data.kdf - data.kmu) / data.kstd,
                scaler={
                    "query_mean": data.gmu.values,
                    "query_std": data.gstd.values,
                    "doc_mean": data.kmu.values,
                    "doc_std": data.kstd.values,
                },
                n_clusters=min(10, len(data.kdf)),
                device=str(device),
            )

            retrieval_stats = evaluate_retrieval(
                retrieval_task, test_dataset, device=str(device)
            )

            # Parseable metric lines
            print(f"\ntop1_accuracy: {retrieval_stats['top1_accuracy']:.4f}")
            print(f"mean_eff: {retrieval_stats['mean_eff']:.4f}")
            print(f"selection_efficiency: {retrieval_stats['selection_efficiency']:.4f}")
            print(f"gmean_eff: {retrieval_stats['gmean_eff']:.4f}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Two Tower tritonBLAS kernel selection model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="two_tower_triton/config/config.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to training data directory (overrides config).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train_eval",
        choices=["train", "eval", "train_eval", "retrieve"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to saved model checkpoint (for eval/retrieve).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config).",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print configuration and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    config = Config(args.config)

    if args.print_config:
        print_config(config)

    # Override config with CLI args
    if args.data_dir:
        config._config.setdefault("io", {})["data_path"] = args.data_dir
    if args.output_dir:
        config._config.setdefault("io", {})["output_path"] = args.output_dir

    main(
        config,
        mode=args.mode,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
    )
