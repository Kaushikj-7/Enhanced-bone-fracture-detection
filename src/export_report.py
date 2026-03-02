import argparse
import glob
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_metrics(output_dir, experiments):
    metrics_data = {}
    for exp in experiments:
        metrics_path = os.path.join(output_dir, exp, f"{exp}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_data[exp] = json.load(f)
    return metrics_data


def create_comparison_charts(metrics_data, output_dir):
    # Extract Accuracy, Precision, Recall, F1 for class '1' (positive) and 'macro avg'
    data = []
    for exp, metrics in metrics_data.items():
        # Weighted avg or macro avg? Let's use macro avg for general performance
        # Or better, the positive class (1) for medical context sensitivity

        # Check if keys are strings "0", "1"
        pos_class = metrics.get("1", {})
        macro_avg = metrics.get("macro avg", {})
        accuracy = metrics.get("accuracy", 0)

        data.append(
            {
                "Experiment": exp,
                "Accuracy": accuracy,
                "F1-Score (Positive)": pos_class.get("f1-score", 0),
                "Recall (Sensitivity)": pos_class.get("recall", 0),
                "Precision": pos_class.get("precision", 0),
            }
        )

    df = pd.DataFrame(data)
    if df.empty:
        return

    # Plot
    df_melted = df.melt("Experiment", var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Experiment", y="Score", hue="Metric")
    plt.title("Model Comparison")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()


def generate_markdown_report(metrics_data, output_dir):
    report_path = os.path.join(output_dir, "final_report.md")
    with open(report_path, "w") as f:
        f.write("# Model Performance Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")

        f.write("## Summary Metrics\n\n")
        f.write(
            "| Experiment | Accuracy | F1 (Pos) | Recall (Pos) | Precision (Pos) |\n"
        )
        f.write("|---|---|---|---|---|\n")

        best_acc = 0
        best_model = "None"

        for exp, metrics in metrics_data.items():
            pos = metrics.get("1", {})
            acc = metrics.get("accuracy", 0)
            f1 = pos.get("f1-score", 0)
            rec = pos.get("recall", 0)
            prec = pos.get("precision", 0)

            f.write(f"| {exp} | {acc:.4f} | {f1:.4f} | {rec:.4f} | {prec:.4f} |\n")

            if acc > best_acc:
                best_acc = acc
                best_model = exp

        f.write("\n")
        f.write(
            f"**Best Performing Model (Accuracy):** {best_model} ({best_acc:.4f})\n\n"
        )

        f.write("## Visualizations\n\n")
        f.write("![Comparison](model_comparison.png)\n\n")

        for exp in metrics_data.keys():
            f.write(f"### {exp}\n\n")
            f.write(f"![ROC](../{exp}/{exp}_roc.png) ")
            f.write(f"![Confusion Matrix](../{exp}/{exp}_cm.png)\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", required=True)
    parser.add_argument("--experiments", default="cnn,vit,hybrid")
    args = parser.parse_args()

    experiments = args.experiments.split(",")

    metrics_data = load_metrics(args.outputs_dir, experiments)

    if not metrics_data:
        print("No metrics found to report.")
        return

    create_comparison_charts(metrics_data, args.outputs_dir)
    generate_markdown_report(metrics_data, args.outputs_dir)
    print(f"Report generated in {args.outputs_dir}/final_report.md")


if __name__ == "__main__":
    main()
