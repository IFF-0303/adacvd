# %%
# %cd '/home/fluebeck/biobank/biobank-llm'
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from accelerate.utils import set_seed
from IPython.display import display
from lifelines import CoxPHFitter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

import pandora.utils.logger
from exploration.evaluation.evaluation import get_evals_path
from exploration.frederike.visualization import color_palette
from exploration.frederike.visualization.visualization_utils import (
    find_evaluation_directories,
    flatten_dict,
    setup_plotting,
)
from pandora.data import ukb_data_utils, ukb_features, ukb_field_ids
from pandora.data.ukb_data_utils import ASSETS_PATH, WANDB_ENTITY, load_ukb_meta_files
from pandora.risk_scores import framingham_risk_score
from pandora.training.dataset import get_column_names, load_prompt_parts, load_split
from pandora.utils.metrics import compute_binary_classification_metrics


def flatten_dict(d, parent_key="", sep="."):
    flattened = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flattened.update(flatten_dict(v, new_key, sep))
        else:
            flattened[new_key] = v
    return flattened


llm_dir = Path(
    "/Users/frederike/Documents/PhD/BioBank/code/2025_03_09_text_data_full/more_runs"
)
# llm_dirs = [
#     Path("/fast/groups/hfm-users/pandora-med-box/results/2025_03_05_text_data_base"),
#     Path(
#         "/fast/groups/hfm-users/pandora-med-box/results/2025_03_10_text_data_full_20k"
#     ),
#     Path(
#         "/fast/groups/hfm-users/pandora-med-box/results/2025_03_10_text_data_full_eos"
#     ),
#     Path(
#         "/fast/groups/hfm-users/pandora-med-box/results/2025_03_10_text_data_full_steps"
#     ),
# ]
llm_metrics = []
for llm_dir in llm_dirs:
    subdirs = sorted([d for d in os.listdir(llm_dir) if d.isdigit() and len(d) == 3])
    print(len(subdirs))

    for subdir in tqdm(subdirs):
        results = {}
        full_path = llm_dir / subdir

        yaml_path = full_path / "text_train_settings.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                text_train_settings = yaml.safe_load(f)
                flat_settings = flatten_dict(text_train_settings)

        files = list(Path(full_path).glob("*evals*.csv"))
        for evals_path in tqdm(files):

            if os.path.exists(evals_path):
                df = pd.read_csv(evals_path)

            metrics = compute_binary_classification_metrics(
                df["y_true"], df["y_pred_score"]
            )

            if len(results) == 0:
                results = metrics.copy()
            else:
                # check if not nan
                if metrics["roc_auc"] is not None:

                    if metrics["roc_auc"] > results["roc_auc"]:
                        results = metrics.copy()
                        # print("Metrics updated")

        resume_training = flat_settings["model.resume_training"]

        results.update(flat_settings)
        results["model"] = (
            "Training from Scratch" if not resume_training else "Adapting Trained Model"
        )
        results["num_training_samples"] = flat_settings["data.num_training_samples"]

        results["subdir"] = subdir
        llm_metrics.append(results.copy())

        # zero-shot
        # find "n0" in filename
        zeroshot = [f for f in files if "n0" in f.name]
        if len(zeroshot) == 0:
            print(f"No zero-shot evals found in {full_path}")
        else:
            evals_path = full_path / zeroshot[0]

            if os.path.exists(evals_path):
                df = pd.read_csv(evals_path)

            metrics = compute_binary_classification_metrics(
                df["y_true"], df["y_pred_score"]
            )

            metrics.update(flat_settings)
            metrics["model"] = (
                "Training from Scratch"
                if not resume_training
                else "Adapting Trained Model"
            )
            metrics["data.num_training_samples"] = 0

            metrics["subdir"] = subdir
            llm_metrics.append(metrics.copy())

llm_metrics = pd.DataFrame(llm_metrics)

llm_metrics = pd.read_parquet(llm_dir / "llm_metrics.parquet")

llm_adaptation_metrics = llm_metrics.loc[
    llm_metrics.groupby(["training.random_seed", "model", "data.num_training_samples"])[
        "roc_auc"
    ].idxmax()
]

llm_adaptation_metrics["model"].value_counts()

assert (
    llm_adaptation_metrics.groupby(["model", "data.num_training_samples"]).size() == 6
).all()


from matplotlib.colors import LinearSegmentedColormap

n = 2
context = "paper"
plt.colormaps.unregister("custom_palette")
sns.set_theme(style="whitegrid")
colors = [
    "#E73EB3",  # pink
    "#874CD6",  # lila
    "#2B0091",  # dark blue
    "#006FF2",  # blue
    "#00D2FF",  # light blue
    "#4AFCE2",  # turquoise
    "#6CFFCA",  # light green
    "#A3FF32",  # green
    "#F9F871",  # yellow
]
cmap = LinearSegmentedColormap.from_list("custom_palette", colors)
plt.colormaps.register(cmap=cmap)
cpal = sns.color_palette("custom_palette", n_colors=n)
sns.set_palette(palette=cpal)

sns.set_context(context, font_scale=1.0)
plt.rcParams["font.family"] = "Helvetica"  # "Arial"
# plt.rcParams["font.size"] = 7
plt.rcParams["xtick.bottom"] = True


llm_adaptation_metrics["resume_training"] = llm_adaptation_metrics[
    "model.resume_training"
].map({True: "Adapting AdaCVD", False: "Fine-Tuning LLM from Scratch"})

# %%


for logscale in [True, False]:
    textwidth = 1.2 * 6.52  # inches
    # Plot results
    w = 0.5 * textwidth
    h = 0.6 * w
    plt.figure(figsize=(w, h))

    # x = "n"
    llm_adaptation_metrics_copy = llm_adaptation_metrics.copy()

    # if logscale:
    #     llm_adaptation_metrics_copy["data.num_training_samples"] = llm_adaptation_metrics[
    #         "data.num_training_samples"
    #     ].replace({0: 1})

    g = sns.lineplot(
        data=llm_adaptation_metrics_copy,
        x="data.num_training_samples",
        y="roc_auc",
        hue="resume_training",
        marker="o",
        errorbar="ci",
        err_style="band",
        palette=[color_palette.TEAL[0], color_palette.GREY[1]],
    )

    FIGURE_PATH = llm_dir / "figures"
    FIGURE_PATH.mkdir(exist_ok=True)

    xlabel = "Number of Training Points"
    plt.xlabel(xlabel)
    plt.ylabel("AUROC" + r" $\longrightarrow$")
    g.get_legend().remove()

    # log-scale x-axis
    if logscale:
        g.set_xscale("symlog", linthresh=10, subs=[2, 3, 4, 5, 6, 7, 8, 9])
        plt.xlim(0, 2 * 10**4 + 10000)

    # Save figure as PDF
    fig_name = "roc_auc_comparison" + ("_log" if logscale else "")
    plt.savefig(FIGURE_PATH / (fig_name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.show()

    fig_legend = plt.figure(figsize=(w, h))
    ax = fig_legend.gca()
    ax.legend(
        *g.get_legend_handles_labels(),
        loc="center",
        frameon=False,
        ncol=3,
        labelspacing=0.5,
        columnspacing=1.0,
    )
    ax.axis("off")
    fig_legend.savefig(
        FIGURE_PATH / (fig_name + "_legend.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    fig_legend.show()

# %%
