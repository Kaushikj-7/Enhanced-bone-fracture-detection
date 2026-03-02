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
    allowed = {"cnn", "vit", "hybrid"}
    for x in experiments:
        if x not in allowed:
            raise argparse.ArgumentTypeError(
                f"Invalid experiment '{x}'. Allowed: cnn, vit, hybrid"
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
    raise ValueError(f"Unknown experiment: {name}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_artifacts", action="store_true")
    parser.add_argument("--skip_report", action="store_true")
    parser.add_argument("--num_gradcam", type=int, default=12)
    args = parser.parse_args()

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
        )
        if len(datasets.get("train", [])) == 0:
            raise SystemExit(
                f"No training data found under {args.data_dir}. Check dataset path."
            )

        pretrained = not args.no_pretrained

        for exp in args.experiments:
            exp_dir = os.path.join(args.output_dir, exp)
            _ensure_dir(exp_dir)

            model = _build_model(exp, args.cnn_backbone, args.vit_backbone, pretrained)
            model = model.to(device)

            config = {
                "dataset": {
                    "root_dir": args.data_dir,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                },
                "model": {
                    "cnn_backbone": args.cnn_backbone,
                    "vit_backbone": args.vit_backbone,
                },
                "training": {
                    "learning_rate": args.learning_rate,
                    "num_epochs": args.epochs,
                    "patience": args.patience,
                },
                "output_dir": exp_dir,
            }

            print("\n" + "=" * 70)
            print("TRAIN:", exp)
            print("=" * 70)
            train_pipeline(
                model=model,
                train_loader=dataloaders["train"],
                val_loader=dataloaders["val"],
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
            ],
            check=True,
            env=env,
        )

    print("\nDone. Outputs in:", args.output_dir)


if __name__ == "__main__":
    main()
