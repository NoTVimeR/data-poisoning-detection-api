from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.attacks import (  # noqa: E402
    inject_complex_subtle_poisoning,
    inject_label_flipping,
    inject_simple_outliers,
    load_real_breast_cancer_dataset,
)
from app.detector import DetectorService  # noqa: E402
from app.evaluation import evaluate_flags  # noqa: E402

OUTPUT = PROJECT_ROOT / "report_assets"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def save_latex(df: pd.DataFrame, filename: str, caption: str, label: str) -> None:
    path = TABLES / filename
    latex = dataframe_to_latex(df, caption, label)
    path.write_text(latex, encoding="utf-8")


def latex_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def dataframe_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    columns = list(df.columns)
    spec = "l" * len(columns)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\hline",
        " & ".join(columns) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_value(row[col]) for col in columns) + " \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def run_real_dataset_results() -> pd.DataFrame:
    detector = DetectorService()
    x, y, _ = load_real_breast_cancer_dataset()
    scenarios = [
        inject_simple_outliers(x, count=50),
        inject_complex_subtle_poisoning(x, count=60),
        inject_label_flipping(x, y, flip_fraction=0.12),
    ]

    rows = []
    for scenario in scenarios:
        if "Label" in scenario.description:
            analysis = detector.analyze_labels(values=scenario.values, labels=scenario.labels)
            predicted = [0] * len(scenario.values)
            for index in analysis["suspicious_indices"]:
                predicted[index] = 1
            rows.append({
                "Attack": "Label Flipping",
                "Method": "KNN Label Consistency",
                **evaluate_flags(scenario.poisoning_labels, predicted),
                "Detected": len(analysis["suspicious_indices"]),
            })
            continue

        attack_name = "Simple Outlier" if "Simple" in scenario.description else "Complex Subtle"
        for method in ["z_score", "isolation_forest", "lof", "hybrid"]:
            analysis = detector.analyze_numeric(values=scenario.values, methods=[method], contamination=0.10)
            predicted = [1 if item["final_flag"] else 0 for item in analysis["results"]]
            rows.append({
                "Attack": attack_name,
                "Method": method.replace("_", " ").title(),
                **evaluate_flags(scenario.poisoning_labels, predicted),
                "Detected": len(analysis["suspicious_indices"]),
            })
    return pd.DataFrame(rows)


def generate_tables(results: pd.DataFrame) -> None:
    f1 = results.pivot_table(index="Attack", columns="Method", values="f1_score", aggfunc="first").reset_index()
    save_latex(
        f1.fillna("-"),
        "T1_f1_method_comparison.tex",
        "Comparison of F1-score across attack scenarios and detection methods.",
        "tab:f1_method_comparison",
    )

    dataset_description = pd.DataFrame([
        {
            "Dataset": "Synthetic simple/complex",
            "Rows": 220,
            "Features": 2,
            "Poisoned samples": 20,
            "Poison ratio": "9.09\\%",
        },
        {
            "Dataset": "Breast Cancer Wisconsin - original",
            "Rows": 569,
            "Features": 30,
            "Poisoned samples": 0,
            "Poison ratio": "0\\%",
        },
        {
            "Dataset": "Breast Cancer + simple outliers",
            "Rows": 619,
            "Features": 30,
            "Poisoned samples": 50,
            "Poison ratio": "8.08\\%",
        },
        {
            "Dataset": "Breast Cancer + subtle poisoning",
            "Rows": 629,
            "Features": 30,
            "Poisoned samples": 60,
            "Poison ratio": "9.54\\%",
        },
        {
            "Dataset": "Breast Cancer + label flipping",
            "Rows": 569,
            "Features": 30,
            "Poisoned samples": 68,
            "Poison ratio": "11.95\\%",
        },
    ])
    save_latex(
        dataset_description,
        "T2_dataset_description.tex",
        "Dataset characteristics used in the experimental evaluation.",
        "tab:dataset_description",
    )

    latency = measure_latency()
    save_latex(
        latency,
        "T3_performance_latency.tex",
        "Average processing latency for different batch sizes using the Z-score endpoint logic.",
        "tab:performance_latency",
    )

    full = results.rename(columns={
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-score",
    })
    save_latex(
        full,
        "T4_full_experimental_results.tex",
        "Detailed experimental results across attacks and methods.",
        "tab:full_experimental_results",
    )


def measure_latency() -> pd.DataFrame:
    detector = DetectorService()
    x, _, _ = load_real_breast_cancer_dataset()
    rows = []
    rng = np.random.default_rng(42)

    for size in [10, 25, 50, 100, 250]:
        indices = rng.choice(len(x), size=size, replace=size > len(x))
        values = x[indices].round(6).tolist()
        timings = []
        for _ in range(8):
            start = time.perf_counter()
            detector.analyze_numeric(values=values, methods=["z_score"])
            timings.append((time.perf_counter() - start) * 1000)
        rows.append({
            "Batch size": size,
            "Method": "Z-score",
            "Average latency (ms)": round(float(np.mean(timings)), 3),
            "Maximum latency (ms)": round(float(np.max(timings)), 3),
        })
    return pd.DataFrame(rows)


def bar_chart(results: pd.DataFrame, attack: str, filename: str, title: str) -> None:
    subset = results[results["Attack"] == attack]
    plt.figure(figsize=(8, 5))
    plt.bar(subset["Method"], subset["f1_score"], color=["#386cb0", "#7fc97f", "#fdc086", "#beaed4", "#f0027f"][:len(subset)])
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=220)
    plt.close()


def architecture_diagram() -> None:
    labels = [
        "Raw Data",
        "FastAPI Service",
        "Preprocessing",
        "Detectors",
        "Hybrid Decision",
        "Cleaned Data",
        "ML Model",
    ]
    plt.figure(figsize=(13, 3.2))
    ax = plt.gca()
    ax.axis("off")
    xs = np.linspace(0.05, 0.95, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef4fb", edgecolor="#2f5f8f", linewidth=1.5),
        )
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.055, 0.5), xytext=(x + 0.055, 0.5), arrowprops=dict(arrowstyle="->", lw=1.6))
    plt.title("API-Based Data Poisoning Detection Pipeline", pad=18)
    plt.tight_layout()
    plt.savefig(FIGURES / "R1_architecture_api_pipeline.jpg", dpi=220)
    plt.close()


def complex_if_outputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, _, _ = load_real_breast_cancer_dataset()
    scenario = inject_complex_subtle_poisoning(x, count=60)
    x_scaled = StandardScaler().fit_transform(np.asarray(scenario.values, dtype=float))
    y_true = np.asarray(scenario.poisoning_labels, dtype=int)
    model = IsolationForest(n_estimators=300, contamination=0.10, random_state=42)
    raw_pred = model.fit_predict(x_scaled)
    y_pred = np.where(raw_pred == -1, 1, 0)
    scores = -model.decision_function(x_scaled)
    return y_true, y_pred, scores


def confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, filename: str, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["Clean", "Poisoned"])
    plt.yticks([0, 1], ["Clean", "Poisoned"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=220)
    plt.close()


def knn_label_confusion() -> None:
    detector = DetectorService()
    x, y, _ = load_real_breast_cancer_dataset()
    scenario = inject_label_flipping(x, y, flip_fraction=0.12)
    analysis = detector.analyze_labels(values=scenario.values, labels=scenario.labels)
    y_pred = np.zeros(len(scenario.values), dtype=int)
    y_pred[analysis["suspicious_indices"]] = 1
    y_true = np.asarray(scenario.poisoning_labels, dtype=int)
    confusion_matrix_plot(
        y_true,
        y_pred,
        "R5_confusion_matrix_knn_label_flipping.jpg",
        "Confusion Matrix - KNN Label Consistency",
    )


def roc_pr_curves(y_true: np.ndarray, scores: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4.5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#386cb0")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Isolation Forest (Complex Attack)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "R6_roc_curve_if_complex.jpg", dpi=220)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6, 4.5))
    plt.plot(recall, precision, color="#7fc97f")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Isolation Forest")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "R7_precision_recall_if_complex.jpg", dpi=220)
    plt.close()


def threshold_tuning_zscore() -> None:
    x, _, _ = load_real_breast_cancer_dataset()
    scenario = inject_complex_subtle_poisoning(x, count=60)
    scaled = StandardScaler().fit_transform(np.asarray(scenario.values, dtype=float))
    scores = np.max(np.abs(scaled), axis=1)
    y_true = np.asarray(scenario.poisoning_labels, dtype=int)
    thresholds = [1.5, 2.0, 2.5, 3.0]
    f1_scores = []
    for threshold in thresholds:
        y_pred = (scores > threshold).astype(int)
        f1_scores.append(evaluate_flags(y_true.tolist(), y_pred.tolist())["f1_score"])

    plt.figure(figsize=(6, 4.5))
    plt.plot(thresholds, f1_scores, marker="o", color="#fdc086")
    plt.xlabel("Z-score threshold")
    plt.ylabel("F1-score")
    plt.title("Threshold Tuning - Z-score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "R8_threshold_tuning_zscore.jpg", dpi=220)
    plt.close()


def generate_figures(results: pd.DataFrame) -> None:
    architecture_diagram()
    bar_chart(
        results,
        "Simple Outlier",
        "R2_method_comparison_simple_outlier.jpg",
        "Method Comparison - Simple Outlier Attack",
    )
    bar_chart(
        results,
        "Complex Subtle",
        "R3_method_comparison_complex_subtle.jpg",
        "Method Comparison - Complex Subtle Attack",
    )
    y_true, y_pred, scores = complex_if_outputs()
    confusion_matrix_plot(
        y_true,
        y_pred,
        "R4_confusion_matrix_if_complex.jpg",
        "Confusion Matrix - Isolation Forest (Complex Attack)",
    )
    knn_label_confusion()
    roc_pr_curves(y_true, scores)
    threshold_tuning_zscore()


def main() -> None:
    ensure_dirs()
    results = run_real_dataset_results()
    results.to_csv(OUTPUT / "report_metrics_source.csv", index=False)
    generate_tables(results)
    generate_figures(results)
    print(f"Report assets generated in: {OUTPUT}")
    print(f"Tables: {TABLES}")
    print(f"Figures: {FIGURES}")


if __name__ == "__main__":
    main()
