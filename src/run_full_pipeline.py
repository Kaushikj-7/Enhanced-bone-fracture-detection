import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import torch

from src.dataset import get_dataloaders
from models.baselines import CNNOnlyModel, ViTOnlyModel
from models.hybrid_model import HybridModel
from training.train import train_pipeline


def _parse_experiments(value: str) -> List[str]:
    experiments = [x.strip() for x in value.split(",") if x.strip()]
    allowed = {"cnn", "vit", "hybrid", "micro"}
    for x in experiments:
        if x not in allowed:
            raise argparse.ArgumentTypeError(
                f"Invalid experiment '{x}'. Allowed: cnn, vit, hybrid, micro"
            )
    return experiments


def _build_model(name: str, cnn_backbone: str, vit_backbone: str, pretrained: bool):
    if name == "cnn":
        return CNNOnlyModel(cnn_backbone=cnn_backbone, pretrained=pretrained)
    if name == "vit":
        return ViTOnlyModel(vit_model=vit_backbone, pretrained=pretrained)
    if name == "hybrid":
        return HybridModel(
            cnn_backbone=cnn_backbone,
            vit_model=vit_backbone,
            pretrained=pretrained,
        )
    if name == "micro":
        from models.micro_hybrid import MicroHybridModel

        return MicroHybridModel(pretrained=pretrained)
    raise ValueError(f"Unknown experiment: {name}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class _LimitedDatasetProxy:
    def __init__(self, size: int):
        self._size = max(1, int(size))

    def __len__(self):
        return self._size


class _LimitedLoader:
    def __init__(self, loader, max_batches: int):
        self.loader = loader
        self.max_batches = max_batches
        batch_size = getattr(loader, "batch_size", 1) or 1
        original_size = len(loader.dataset)
        limited_size = min(original_size, max_batches * batch_size)
        self.dataset = _LimitedDatasetProxy(limited_size)

    def __iter__(self):
        for idx, batch in enumerate(self.loader):
            if idx >= self.max_batches:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.max_batches)


def _maybe_limit_loader(loader, max_batches: int):
    if not loader or max_batches <= 0:
        return loader
    return _LimitedLoader(loader, max_batches)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline: train → artifacts (ROC/CM/Grad-CAM) → final report"
    )
    parser.add_argument("--data_dir", default="data", help="Dataset root")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--experiments",
        type=_parse_experiments,
        default=["cnn", "vit", "hybrid"],
        help="Comma-separated list: cnn,vit,hybrid",
    )
    parser.add_argument("--cnn_backbone", default="resnet18")
    parser.add_argument("--vit_backbone", default="vit_base_patch16_224")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation")
    parser.add_argument("--low_resource", action="store_true", help="Shortcut for low-memory environments (B8, Acc4, SimplePre)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"], help="Optimizer choice")
    parser.add_argument("--loss", default="bce", choices=["bce", "focal"], help="Loss function choice")
    parser.add_argument("--simple_pre", action="store_true", help="Use minimal preprocessing (CLAHE only)")
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="Limit train batches per epoch for fast validation runs (0 = full)",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=0,
        help="Limit val batches per epoch for fast validation runs (0 = full)",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help="Limit artifact eval batches per split (0 = full)",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Unfreeze the last blocks of backbones for better performance.",
    )
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_artifacts", action="store_true")
    parser.add_argument("--skip_report", action="store_true")
    parser.add_argument("--num_gradcam", type=int, default=12)
    parser.add_argument(
        "--load_model", default=None, help="Path to existing model checkpoint (.pth)"
    )
    parser.add_argument(
        "--eval_splits",
        default="val,test",
        help="Comma-separated eval splits for artifacts/report, e.g. val or val,test",
    )
    args = parser.parse_args()

    # Apply Low Resource Overrides
    if args.low_resource:
        print(">>> Applying LOW RESOURCE mode (Batch=8, Accumulation=4, SimplePre=True)")
        args.batch_size = 8
        args.accumulation_steps = 4
        args.simple_pre = True
        if args.num_workers > 0:
            args.num_workers = 1 # Reduce CPU overhead

    _ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Experiments:", args.experiments)
    print("Output dir:", args.output_dir)

    # Training
    if not args.skip_train:
        dataloaders, datasets = get_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            simple_pre=args.simple_pre,
        )
        if len(datasets.get("train", [])) == 0:
            raise SystemExit(
                f"No training data found under {args.data_dir}. Check dataset path."
            )

        train_loader = _maybe_limit_loader(dataloaders["train"], args.max_train_batches)
        val_loader = _maybe_limit_loader(dataloaders["val"], args.max_val_batches)

        pretrained = not args.no_pretrained

        for exp in args.experiments:
            exp_dir = os.path.join(args.output_dir, exp)
            _ensure_dir(exp_dir)

            model = _build_model(exp, args.cnn_backbone, args.vit_backbone, pretrained)

            # Load existing checkpoint if provided
            if args.load_model and os.path.exists(args.load_model):
                print(f"Loading weights from {args.load_model} for {exp}...")
                try:
                    # Map location to CPU first to avoid CUDA errors, then move to device
                    state_dict = torch.load(args.load_model, map_location="cpu")
                    model.load_state_dict(state_dict)
                    print("Checkpoint loaded successfully.")
                except Exception as e:
                    print(f"Warning: Could not load checkpoint: {e}")

            model = model.to(device)

            # Support transfer learning for micro model (freeze CNN backbone)
            if exp == "micro" and not args.fine_tune:
                if hasattr(model, "freeze_backbone"):
                    model.freeze_backbone()

            # Apply fine-tuning if requested (unfreeze last blocks)
            if args.fine_tune and hasattr(model, "set_fine_tuning"):
                print(f"Enabling fine-tuning for {exp} experiment...")
                model.set_fine_tuning(True, True)

                # If using default LR (0.001), lower it for fine-tuning to prevent catastrophic forgetting
                if args.learning_rate == 0.001:
                    print("Lowering default learning rate to 0.0001 for fine-tuning.")
                    args.learning_rate = 0.0001

            config = {
                "dataset": {
                    "root_dir": args.data_dir,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "simple_pre": args.simple_pre,
                },
                "model": {
                    "cnn_backbone": args.cnn_backbone,
                    "vit_backbone": args.vit_backbone,
                },
                "training": {
                    "learning_rate": args.learning_rate,
                    "num_epochs": args.epochs,
                    "patience": args.patience,
                    "optimizer": args.optimizer,
                    "loss": args.loss,
                    "accumulation_steps": args.accumulation_steps,
                },
                "output_dir": exp_dir,
            }

            print("\n" + "=" * 70)
            print("TRAIN:", exp)
            print("=" * 70)
            train_pipeline(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
                save_dir=exp_dir,
            )

    # Artifacts + report are invoked as module CLIs to avoid duplicating logic.
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    if not args.skip_artifacts:
        print("\n" + "=" * 70)
        print("ARTIFACTS: ROC/CM + Grad-CAM")
        print("=" * 70)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.finalize_artifacts",
                "--data_dir",
                args.data_dir,
                "--output_dir",
                args.output_dir,
                "--experiments",
                ",".join(args.experiments),
                "--cnn_backbone",
                args.cnn_backbone,
                "--vit_backbone",
                args.vit_backbone,
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--num_gradcam",
                str(args.num_gradcam),
                "--splits",
                args.eval_splits,
                "--max_eval_batches",
                str(args.max_eval_batches),
            ],
            check=True,
            env=env,
        )

    if not args.skip_report:
        print("\n" + "=" * 70)
        print("REPORT: final_report.md + CSV + plot")
        print("=" * 70)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.export_report",
                "--outputs_dir",
                args.output_dir,
                "--experiments",
                ",".join(args.experiments),
                "--splits",
                args.eval_splits,
            ],
            check=True,
            env=env,
        )

    print("\nDone. Outputs in:", args.output_dir)


if __name__ == "__main__":
    main()
