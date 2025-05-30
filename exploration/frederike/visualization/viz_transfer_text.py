import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from exploration.frederike.visualization.visualization_utils import (
    find_prediction_directories,
)
from pandora.utils.metrics import compute_binary_classification_metrics

textwidth = 1.2 * 6.52  # inches

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


BASE_PATH = Path(
    "/Users/frederike/Documents/PhD/BioBank/code/2025_03_03_text_data_full/"
)

# Define directories
PRED_PATHS = {
    "training": BASE_PATH / "training",
    "adaptation": BASE_PATH / "adaptation",
}

FIGURE_PATH = BASE_PATH / "figures"
FIGURE_PATH.mkdir(exist_ok=True)

# Define regex pattern to extract epoch and n
filename_pattern = re.compile(r"evals_(\d+)_n(\d+)\.csv")


# Function to read CSV files and compute metrics
def process_directory(base_dir, label):
    results = []

    dirs = find_prediction_directories(base_dir)
    for directory in dirs:
        # read text_train_setting.yaml
        with open(directory / "text_train_settings.yaml", "r") as f:
            settings = yaml.safe_load(f)

        for filename in tqdm(os.listdir(directory)):
            match = filename_pattern.match(filename)
            if match:
                epoch, n = map(int, match.groups())
                file_path = os.path.join(directory, filename)

                evals = pd.read_csv(file_path)
                metrics = compute_binary_classification_metrics(
                    evals["y_true"], evals["y_pred_score"]
                )

                results.append(
                    {
                        "epoch": epoch,
                        "n": n,
                        "roc_auc": metrics["roc_auc"],
                        "model": label,
                        "eval_steps": settings["training"]["eval_steps"],
                        "random_seed": settings["training"]["random_seed"],
                        "num_gpus": settings["condor"]["request_gpus"],
                    }
                )

    return pd.DataFrame(results)


df_0 = process_directory(PRED_PATHS["training"], "Fine-Tuning LLM From Scratch")
df_1 = process_directory(PRED_PATHS["adaptation"], "Adapting AdaCVD")

# Combine results
df = pd.concat([df_0, df_1], ignore_index=True)

# Sort values for visualization
df = df.sort_values(by=["epoch", "n"])

df["min_epoch"] = df.groupby("model")["epoch"].transform("min")

if "base" in BASE_PATH.name:
    df = df[df["epoch"] == df["min_epoch"]]
elif "full" in BASE_PATH.name:
    df = df[df["epoch"] <= df["min_epoch"]]

df["eval_step"] = (
    df.groupby(["model", "random_seed"]).cumcount()
    * df["eval_steps"]
    * df["num_gpus"]
    / 2
)

# Plot results
w = 0.5 * textwidth
h = 0.6 * w
plt.figure(figsize=(w, h))

x = "eval_step"
# x = "n"

g = sns.lineplot(
    data=df,
    x=x,
    y="roc_auc",
    hue="model",
    marker="o",
    errorbar="pi",
    err_style="band",
    hue_order=["Adapting AdaCVD", "Fine-Tuning LLM From Scratch"],
)

xlabel = "Number of Training Steps" if x == "eval_step" else "Number of Training Points"
plt.xlabel(xlabel)
plt.ylabel("AUROC")
# plt.title("Comparison of Fine-Tuning LLM From Scratch vs. Adapting a Trained Model")
g.get_legend().remove()

# Save figure as PDF
fig_name = "roc_auc_comparison"
plt.savefig(FIGURE_PATH / (fig_name + ".pdf"), format="pdf", bbox_inches="tight")
plt.show()

fig_legend = plt.figure(figsize=(w, h))
ax = fig_legend.gca()
ax.legend(
    *g.get_legend_handles_labels(),
    loc="center",
    frameon=False,
    ncol=len(g.get_legend_handles_labels()[1]),
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
