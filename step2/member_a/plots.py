"""Figures generated from Step 2A aggregate tables."""

import matplotlib

matplotlib.use("Agg")  # Run on servers and in CI without opening a window.
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


METHODS = ("linear_probe", "lora_r8", "lora_r16")
NAMES = {"linear_probe": "Linear probe", "lora_r8": "LoRA-r8", "lora_r16": "LoRA-r16", "pretrained": "Single-template baseline"}
COLORS = {"linear_probe": "#136c92", "lora_r8": "#be5319", "lora_r16": "#396c4e", "pretrained": "#6c6575"}
MARKERS = {"linear_probe": "o", "lora_r8": "s", "lora_r16": "^"}
DOMAINS = {"mimic": "MIMIC (source)", "chexpert": "CheXpert (target)"}
METRICS = {"macro_auc": "Macro ROC-AUC", "macro_f1": "Macro F1", "macro_auprc": "Macro AUPRC"}
BUDGETS = ("10pct", "100pct")
CLASSES = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")
SHORT_NAMES = ("Atelectasis", "Cardio-\nmegaly", "Consoli-\ndation", "Edema", "Pleural\neffusion")


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e3e8ed", linewidth=0.8)


def save(figure, path, footnote, bottom=0.09):
    figure.text(0.02, 0.016, footnote, fontsize=9, color="#475569")
    figure.tight_layout(rect=(0, bottom, 1, 0.96))
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def budget_curves(macro, path):
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 6.7), sharex=True, sharey=True)
    figure.suptitle("Classification performance by source-label budget", fontsize=16, fontweight="bold")
    x = [0, 1]
    for row, (domain, label) in enumerate(DOMAINS.items()):
        values = macro.query("domain == @domain").set_index(["method", "budget"])
        for column, (metric, name) in enumerate(METRICS.items()):
            axis = axes[row, column]
            for method in METHODS:
                runs = values.loc[method].reindex(BUDGETS)
                axis.errorbar(x, runs[f"{metric}_mean"], yerr=runs[f"{metric}_std"], marker=MARKERS[method],
                              capsize=3, color=COLORS[method], label=NAMES[method])
            baseline = values.loc[("pretrained", "100pct"), f"{metric}_mean"]
            axis.axhline(baseline, color=COLORS["pretrained"], linestyle="--", label=NAMES["pretrained"])
            if row == 0:
                axis.set_title(name)
            if column == 0:
                axis.set_ylabel(label)
            axis.set_ylim(0, 1.03)
            axis.set_xticks(x, ["10%", "100%"])
            axis.set_xlim(-0.17, 1.17)
            style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=4, frameon=False)
    save(figure, path, "Points and error bars show mean ± sample SD over seeds 42, 43 and 44. The frozen baseline is fixed.", 0.09)


def transfer_drops(transfer, macro, path):
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.9))
    figure.suptitle("Signed change from source test to target test", fontsize=16, fontweight="bold")
    by_method = transfer.set_index(["method", "budget"])
    x = np.arange(len(BUDGETS))
    for axis, (metric, name) in zip(axes, METRICS.items()):
        field = f"{metric}_drop_source_minus_target"
        axis.axhline(0, color="#17233a", linewidth=0.9)
        baseline = by_method.loc[("pretrained", "100pct"), f"{field}_mean"]
        axis.axhline(baseline, color=COLORS["pretrained"], linestyle="--", label=NAMES["pretrained"])
        for offset, method in zip((-0.25, 0, 0.25), METHODS):
            runs = by_method.loc[method].reindex(BUDGETS)
            axis.bar(x + offset, runs[f"{field}_mean"], yerr=runs[f"{field}_std"], width=0.23,
                     capsize=3, color=COLORS[method], label=NAMES[method])
        axis.set_xticks(x, ["10%", "100%"])
        axis.set_title(name)
        axis.set_xlabel("Source-label budget")
        style(axis)
    axes[0].set_ylabel("MIMIC − CheXpert")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=4, frameon=False)
    counts = {
        domain: ", ".join(map(str, sorted(macro.loc[macro["domain"] == domain, "macro_auc_valid_classes"].unique())))
        for domain in DOMAINS
    }
    save(figure, path,
         f"Bars show mean ± sample SD of seed-matched gaps. Main ROC-AUC uses {counts['mimic']} source classes and {counts['chexpert']} target classes.",
         0.13)


def disease_profiles(per_class, prevalence, budget, path):
    figure, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True)
    percent = budget.removesuffix("pct")
    figure.suptitle(f"Performance and prevalence by disease · {percent}% source-label budget",
                    fontsize=16, fontweight="bold")
    x = np.arange(len(CLASSES))
    f1 = per_class.loc[(per_class["budget"] == budget) | (per_class["method"] == "pretrained")]
    f1_lower = min(-0.025, (f1["f1_mean"] - f1["f1_std"].fillna(0)).min() - 0.025)
    for column, (domain, label) in enumerate(DOMAINS.items()):
        values = per_class.query("domain == @domain").set_index(["method", "budget", "class_name"])
        for row, (metric, name) in enumerate((("auc", "ROC-AUC"), ("f1", "F1"))):
            axis = axes[row, column]
            for offset, method in zip((-0.25, -0.08, 0.08, 0.25), (*METHODS, "pretrained")):
                method_budget = "100pct" if method == "pretrained" else budget
                runs = values.loc[(method, method_budget)].reindex(CLASSES)
                axis.errorbar(x + offset, runs[f"{metric}_mean"],
                              yerr=None if method == "pretrained" else runs[f"{metric}_std"],
                              fmt="o", markersize=4, capsize=2, color=COLORS[method], label=NAMES[method])
            axis.set_ylim(-0.025 if metric == "auc" else f1_lower, 1.05)
            if metric == "auc":
                for position, disease in enumerate(CLASSES):
                    if values.xs(disease, level="class_name")["auc_mean"].isna().all():
                        axis.text(position, 0.48, "NA", ha="center", color="#64748b", fontsize=9)
            axis.set_ylabel(name)
            style(axis)
        rates = prevalence.query("domain == @domain").set_index("class_name").reindex(CLASSES)
        prevalence_axis = axes[2, column]
        prevalence_axis.bar(x, rates["positive_rate"], width=0.5, color="#8c9aaf")
        prevalence_axis.set_ylim(0, 1.0)
        prevalence_axis.yaxis.set_major_formatter(PercentFormatter(1))
        prevalence_axis.set_ylabel("Positive rate")
        prevalence_axis.set_xticks(x, SHORT_NAMES)
        axes[0, column].set_title(label)
        style(prevalence_axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=4, frameon=False)
    save(figure, path, "Positive rate uses only valid labels for each disease. Error bars show sample SD over three seeds.", 0.08)


def write_plots(macro, transfer, per_class, prevalence, folder):
    folder.mkdir(parents=True, exist_ok=True)
    budget_curves(macro, folder / "budget_curves.png")
    transfer_drops(transfer, macro, folder / "transfer_drops.png")
    for budget in BUDGETS:
        disease_profiles(per_class, prevalence, budget, folder / f"disease_profiles_{budget}.png")
