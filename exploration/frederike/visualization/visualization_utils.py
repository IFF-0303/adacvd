import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm


def flatten_dict(d, parent_key="", sep="."):
    """Flatten a nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def find_evaluation_directories(eval_path):
    """Find all directories that contain the following three files: settings.yaml, metrics.csv, roc_curve_values.yaml"""
    required_files = {"settings.yaml", "metrics.csv", "roc_curve_values.yaml"}
    evaluation_directories = []

    for root, dirs, files in os.walk(eval_path):
        if required_files.issubset(files):
            evaluation_directories.append(Path(root))

    return evaluation_directories


def find_prediction_directories(eval_path):
    required_files = {"text_train_settings.yaml"}
    prediction_directories = []

    for root, dirs, files in os.walk(eval_path):
        if required_files.issubset(files):
            prediction_directories.append(Path(root))

    return prediction_directories


def setup_plotting(context="talk", n=28):
    plt.colormaps.unregister("custom_palette")
    sns.set_theme(style="whitegrid", font="serif")
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
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 14
    # sns.set_context(context, font_scale=1.2)
    return cpal


def retrieve_results_from_path(eval_dirs):
    metrics_df = pd.DataFrame()
    settings = pd.DataFrame()
    roc_curve_values = {}
    evals = pd.DataFrame()

    for dir in tqdm(eval_dirs):
        file = dir / "metrics.csv"
        metrics_tmp = pd.read_csv(file, index_col=0)
        metrics_tmp.index.name = "bootstrap_round"

        metrics_tmp["directory"] = str(dir)
        metrics_df = pd.concat([metrics_df, metrics_tmp.reset_index()], axis=0)

        with open(dir / "settings.yaml", "r") as f:
            settings_tmp_d = yaml.safe_load(f)

        settings_tmp = pd.DataFrame([flatten_dict(settings_tmp_d)])
        settings_tmp["directory"] = str(dir)
        settings = pd.concat([settings, settings_tmp], axis=0)

        with open(dir / "roc_curve_values.yaml", "r") as f:
            roc_curve_values_tmp = yaml.safe_load(f)
        roc_curve_values[str(dir)] = roc_curve_values_tmp

        evals_tmp = pd.read_csv(dir / "evals_subset.csv", index_col=0)
        evals_tmp["directory"] = str(dir)
        evals = pd.concat([evals, evals_tmp], axis=0)

    return metrics_df, settings, roc_curve_values, evals


def retrieve_results_from_groups(eval_dirs: dict, mode: str = "full"):
    metrics = pd.DataFrame()
    settings = pd.DataFrame()
    roc_curve_values = {}
    evals = pd.DataFrame()

    for name, path in eval_dirs.items():
        print(f"Loading evaluation results from: {name}")
        all_eval_dirs = find_evaluation_directories(path)
        eval_dirs = [x for x in all_eval_dirs if mode in x.parts]

        metrics_df_tmp, settings_tmp, roc_curve_values_tmp, evals_tmp = (
            retrieve_results_from_path(eval_dirs)
        )

        metrics_df_tmp["group"] = name
        evals_tmp["group"] = name

        metrics = pd.concat([metrics, metrics_df_tmp], axis=0)
        settings = pd.concat([settings, settings_tmp], axis=0)
        evals = pd.concat([evals, evals_tmp], axis=0)
        roc_curve_values.update(roc_curve_values_tmp)

    plot_df = metrics.merge(
        settings, left_on="directory", right_on="directory", how="left"
    )

    return plot_df, roc_curve_values, evals
