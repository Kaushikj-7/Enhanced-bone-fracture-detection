import argparse
import glob
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_metrics(output_dir, experiments, splits):
    metrics_data = {}
    for exp in experiments:
        metrics_data[exp] = {}
        for split_name in splits:
            split_metrics = os.path.join(
                output_dir, exp, f"{exp}_{split_name}_metrics.json"
            )
            legacy_metrics = os.path.join(output_dir, exp, f"{exp}_metrics.json")

            if os.path.exists(split_metrics):
                with open(split_metrics, "r") as f:
                    metrics_data[exp][split_name] = json.load(f)
            elif split_name == "val" and os.path.exists(legacy_metrics):
                with open(legacy_metrics, "r") as f:
                    metrics_data[exp][split_name] = json.load(f)

        if len(metrics_data[exp]) == 0:
            del metrics_data[exp]
    return metrics_data


def create_comparison_charts(metrics_data, output_dir):
    # Extract Accuracy, Precision, Recall, F1 for class '1' (positive) and 'macro avg'
    split_rows = {}
    for exp, split_map in metrics_data.items():
        for split_name, metrics in split_map.items():
            pos_class = metrics.get("1", {})
            accuracy = metrics.get("accuracy", 0)
            split_rows.setdefault(split_name, []).append(
                {
                    "Experiment": exp,
                    "Accuracy": accuracy,
                    "F1-Score (Positive)": pos_class.get("f1-score", 0),
                    "Recall (Sensitivity)": pos_class.get("recall", 0),
                    "Precision": pos_class.get("precision", 0),
                }
            )

    for split_name, rows in split_rows.items():
        df = pd.DataFrame(rows)
        if df.empty:
            continue

        df_melted = df.melt("Experiment", var_name="Metric", value_name="Score")

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melted, x="Experiment", y="Score", hue="Metric")
        plt.title(f"Model Comparison [{split_name}]")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"model_comparison_{split_name}.png"))
        plt.close()


def generate_markdown_report(metrics_data, output_dir):
    report_path = os.path.join(output_dir, "final_report.md")
    with open(report_path, "w") as f:
        f.write("# Model Performance Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")

        f.write("## Summary Metrics\n\n")
        f.write(
            "| Split | Experiment | Accuracy | F1 (Pos) | Recall (Pos) | Precision (Pos) |\n"
        )
        f.write("|---|---|---|---|---|---|\n")

        best_acc = 0
        best_model = "None"
        best_split = "N/A"

        for exp, split_map in metrics_data.items():
            for split_name, metrics in split_map.items():
                pos = metrics.get("1", {})
                acc = metrics.get("accuracy", 0)
                f1 = pos.get("f1-score", 0)
                rec = pos.get("recall", 0)
                prec = pos.get("precision", 0)

                f.write(
                    f"| {split_name} | {exp} | {acc:.4f} | {f1:.4f} | {rec:.4f} | {prec:.4f} |\n"
                )

                if acc > best_acc:
                    best_acc = acc
                    best_model = exp
                    best_split = split_name

        f.write("\n")
        f.write(
            f"**Best Performing Model (Accuracy):** {best_model} [{best_split}] ({best_acc:.4f})\n\n"
        )

        f.write("## Visualizations\n\n")
        split_names = sorted(
            {
                split_name
                for split_map in metrics_data.values()
                for split_name in split_map
            }
        )
        for split_name in split_names:
            f.write(f"### Comparison [{split_name}]\n\n")
            f.write(f"![Comparison](model_comparison_{split_name}.png)\n\n")

        for exp, split_map in metrics_data.items():
            f.write(f"### {exp}\n\n")
            for split_name in split_map.keys():
                f.write(f"#### {split_name}\n\n")
                f.write(f"![ROC](../{exp}/{exp}_{split_name}_roc.png) ")
                f.write(f"![Confusion Matrix](../{exp}/{exp}_{split_name}_cm.png)\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", required=True)
    parser.add_argument("--experiments", default="cnn,vit,hybrid")
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated splits to include in report, e.g. val or val,test",
    )
    args = parser.parse_args()

    experiments = args.experiments.split(",")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    metrics_data = load_metrics(args.outputs_dir, experiments, splits)

    if not metrics_data:
        print("No metrics found to report.")
        return

    create_comparison_charts(metrics_data, args.outputs_dir)
    generate_markdown_report(metrics_data, args.outputs_dir)
    print(f"Report generated in {args.outputs_dir}/final_report.md")


if __name__ == "__main__":
    main()
