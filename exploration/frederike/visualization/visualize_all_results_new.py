# %%

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from exploration.frederike.visualization import color_palette, name_mapping
from exploration.frederike.visualization.visualization_utils import (
    find_evaluation_directories,
    flatten_dict,
    retrieve_results_from_groups,
    setup_plotting,
)
from pandora.data import ukb_data_utils, ukb_features, ukb_field_ids
from pandora.utils.metrics import (
    compute_binary_classification_metrics,
    plot_roc_auc_curve,
)

# %%

# ------------------------ SPECIFY EVALUATION RESULTS ------------------------ #
save = True

FIGURE_PATH = Path(
    "//Users/frederike/Documents/PhD/BioBank/code/2025_03_10_report/figures"
)
FIGURE_PATH.mkdir(exist_ok=True)

EVAL_PATHS = {
    # "cox_ph_baseline": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_01_16_report/2025_01_26_ml_baselines_sweeps/cox"
    # ),
    # "lgbm": Path("//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/lgbm"),
    # "logreg_baseline": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_01_16_report/2025_01_26_ml_baselines_sweeps/logreg"
    # ),
    # "risk_scores": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/risk_scores"
    # ),
    # "different_base_llms": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_01_16_report/different_LLMs"
    # ),
    # # "base_llm_mistral": Path(
    # #     "//Users/frederike/Documents/PhD/BioBank/code/2025_01_16_report/base_risk_factors/2025_01_16_base_risk_factors_extended_target/001/"
    # # ), # <--- included below (separate models)
    # "base_llm_mistral": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/mistral_separate_models/000"
    # ),
    # "mistral_separate_models": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/mistral_separate_models"
    # ),
    ##
    "cox_ph_baseline": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_19_report/cox"
    ),
    "lgbm": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_28_report/lgbm_all_fgs_params/001"
    ),
    "logreg_baseline": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_03_10_report/logreg"
    ),
    "risk_scores": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_19_report/risk_scores"
    ),
    # "different_base_llms": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_19_report/different_base_models"
    # ),
    "base_llm_mistral": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_03_10_report/mistral_more_fgs/001"
    ),
    "base_llm_llama_8b": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_28_report/base_mistral_llama/003"
    ),
    "base_llm_llama_3b": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_19_report/different_base_models/001"
    ),
    "base_llm_phi": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_19_report/different_base_models/003"
    ),
    "base_llm_gemma": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_19_report/different_base_models/005"
    ),
    # "mistral_separate_models": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_19_report/mistral_fgs"
    # ),
    # "zeroshot": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_28_report/zeroshot"
    # ),
    "zeroshot_mistral": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_28_report/zeroshot/005"
    ),
    "zeroshot_llama": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_28_report/zeroshot/000"
    ),
    "mistral_all_fgs": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_03_10_report/mistral_more_fgs"
    ),
    "mistral_full_model": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_03_10_report/full_model"
    ),
    "mistral_flexible_model": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_03_10_report/flexible_model"
    ),
    # "lgbm_all_fgs": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_28_report/lgbm_all_fgs"
    # ),
    "lgbm_all_fgs_rounded": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_28_report/lgbm_all_fgs_rounded"
    ),
    "lgbm_all_fgs_params": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_28_report/lgbm_all_fgs_params"
    ),
    ##
    # "format_inference": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_19_format_inference_report/transformed"
    # ),
    # "format_inference_orig": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_19_format_inference_report/orig"
    # ),
    ##
    # "base_mistral_oversampling": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/base_mistral_oversampling"
    # ),
    # "base_llm_mistral_exact_sets": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_01_16_report/base_risk_factors/2025_01_16_base_risk_factors_extended_target_exact_sets"
    # ),
    # "data_efficiency_lgbm": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_01_16_report/data_efficiency_lgbm"
    # ),
    # "data_efficiency_mistral": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_01_16_report/data_efficiency_mistral"
    # ),
    # "cox_data_efficiency": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_01_16_report/cox_data_efficiency"
    # ),
    # "cosine": Path("//Users/frederike/Documents/PhD/BioBank/code/2025_01_23_misc"),
    # "lgbm_all_combs": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_01_16_report/lgbm_more_risk_factors"
    # ),
    # "separate_models": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_01_16_report/separate_models"
    # ),
    # "flexible_model": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_01_16_report/flexible_model"
    # ),
    # "separate_models_2": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/full_models/2025_01_31_full_model_r/"
    # ),
    # "full_model": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/full_model_inference"
    # ),
    # "flexible_model_new": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/flexible_model"
    # ),
    # "flexible_model": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/flexible_model_new/2025_02_04_flexible_model_from_full_model/model_000/"  # new_new
    # ),
    # LGBM
    "lgbm_flexible_model": Path(
        "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/lgbm_flexible_model/median"
    ),
    # "lgbm_more_risk_factors": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/lgbm_more_risk_factors"
    # ),
    # "lgbm_more_risk_factors_2": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/lgbm_more_risk_factors_2"
    # ),
    # "lgbm_more_risk_factors_3": Path(
    #     "//Users/frederike/Documents/PhD/BioBank/code/2025_02_01_report/lgbm_more_risk_factors_3"
    # ),
    # "lgbm_transfer": Path(
    #     "/Users/frederike/Documents/PhD/BioBank/code/2025_02_11_transfer_eval/2025_02_11_eval_transfer/lgbm_all_subgroups"
    # ),
    # "llm_transfer": Path(
    #     "/Users/frederike/Documents/PhD/BioBank/code/2025_02_11_transfer_eval/2025_02_11_eval_transfer/llm"
    # ),
    # "lgbm_transfer_sample": Path(
    #     "/Users/frederike/Documents/PhD/BioBank/code/2025_02_11_transfer_eval/2025_02_11_eval_transfer/lgbm_sample"
    # ),
    # "lgbm": Path(
    #     "/Users/frederike/Documents/PhD/BioBank/code/2025_02_11_transfer_eval/2025_02_11_eval_transfer/lgbm_all"
    # ),
    #     "lgbm_full_model": Path(
    #         "/Users/frederike/Documents/PhD/BioBank/code/2025_02_11_transfer_eval/2025_02_11_eval_transfer/lgbm_full_model"
    #     ),
}
# %%

# LOAD FULL EVALUATION RESULTS
plot_df, roc_curve_values, evals = retrieve_results_from_groups(EVAL_PATHS, mode="full")

# %%
# LOAD SUBGROUP EVALUATION RESULTS
load_subgroups = True
if load_subgroups:
    plot_df_sg, roc_curve_values_sg, evals_sg = retrieve_results_from_groups(
        EVAL_PATHS, mode="subgroups"
    )


# %%

# ----------------------------------------------------------------- #
# ------------------------ PROCESS COLUMNS ------------------------ #
# ----------------------------------------------------------------- #

# ------------------------ Base Risk Factors ------------------------ #

base_risk_factors_framingham = ["base_risk_score_inputs"]
base_risk_factors_aha_aca = [
    "base_risk_score_inputs",
    "additional_risk_score_inputs_aha_acc",
]
base_risk_factors_prevent = [
    "base_risk_score_inputs",
    "additional_risk_score_inputs_prevent",
]
base_risk_factors = [
    "base_risk_score_inputs",
    "additional_risk_score_inputs_aha_acc",
    "additional_risk_score_inputs_prevent",
]


# ------------------------ MAP MODELS TO NAMES ------------------------ #
def get_model_name(df):
    df["model_name"] = None

    group_to_name_mapping = {
        "risk_scores": lambda df: df["model"],
        "lgbm": lambda _: "LGBM",  # "Gradient Boosted Trees",
        "lgbm_full_model": lambda _: "LGBM (full model)",  # "Gradient Boosted Trees",
        "lgbm_baseline_more_features": lambda _: "Gradient Boosted Trees",
        "lgbm_all_fgs": lambda _: "LGBM",
        "lgbm_all_fgs_rounded": lambda _: "LGBM",
        "lgbm_all_fgs_params": lambda _: "LGBM",
        "lgbm_flexible_model": lambda _: "Full Model (LGBM)",
        "lgbm_more_risk_factors": lambda _: "Separate Models (LGBM)",
        "logreg_baseline": lambda _: "Logistic Regression",
        "cox_ph_baseline": lambda _: "Cox PH Model",
        "different_base_llms": lambda df: "Fine-Tuned LLM " + df["model.name"],
        "base_llm_mistral": lambda df: "Fine-Tuned LLM " + df["model.name"],
        "base_llm_llama_8b": lambda df: "Fine-Tuned LLM " + df["model.name"],
        "base_llm_llama_3b": lambda df: "Fine-Tuned LLM " + df["model.name"],
        "base_llm_phi": lambda df: "Fine-Tuned LLM " + df["model.name"],
        "base_llm_gemma": lambda df: "Fine-Tuned LLM " + df["model.name"],
        "zeroshot": lambda df: "Zero-Shot LLM " + df["model.name"],
        "zeroshot_mistral": lambda df: "Zero-Shot LLM " + df["model.name"],
        "zeroshot_llama": lambda df: "Zero-Shot LLM " + df["model.name"],
        "mistral_separate_models": lambda df: "Fine-Tuned LLM " + df["model.name"],
        "mistral_all_fgs": lambda df: "Fine-Tuned LLM " + df["model.name"],
        "base_llm_mistral_exact_sets": lambda df: "Fine-Tuned LLM " + df["model.name"],
        "data_efficiency_lgbm": lambda df: "Gradient Boosted Trees",
        "data_efficiency_mistral": lambda df: "Fine-Tuned LLM " + df["model.name"],
        "cox_data_efficiency": lambda df: "Cox PH Model",
        # ...
        "zeroshot": lambda df: "Zero-Shot " + df["model.name"],
        "mistral_flexible_model": lambda df: "Flexible Model (LLM)",  # + df["model.name"],
        "mistral_full_model": lambda df: "Full Model (LLM)",  # + df["model.name"],
        "flexible_model_new": lambda df: "Flexible Model New" + df["model.name"],
        "separate_models": lambda df: "Separate Models (LLM)",  # + df["model.name"],
        # "mistral_separate_models": lambda df: "Separate Models (LLM)",  # + df["model.name"],
        "separate_models_2": lambda df: "Separate Models " + df["model.name"],
        "flexible_sampling": lambda df: (
            df["directory"].str.extract(r"(model_\d+)").squeeze()
            + "_"
            + df["directory"].str.extract(r"(epoch_\d+)").squeeze()
        ),
        "flexible_sampling_new_combs": lambda df: (
            df["directory"].str.extract(r"(model_\d+)").squeeze()
            + "_"
            + df["directory"].str.extract(r"(epoch_\d+)").squeeze()
        ),
        "flexible_sampling_new_models": lambda df: (
            "New " + df["directory"].str.extract(r"(model_\d+)").squeeze()
        ),
        "flexible_model_new": lambda df: (
            "New II " + df["directory"].str.extract(r"(model_\d+)").squeeze()
        ),
        # "separate_models": lambda _: "Separate Models New",
        "separate_models_MH": lambda _: "Separate Models MH",
        "separate_models_MH_2": lambda _: "Separate Models MH 2",
        "separate_models_same_subset": lambda _: "Separate Models Same Subset",
        "flexible_sampling_improved_more_combs": lambda df: (
            "Improved Sampling " + df["directory"].str.extract(r"(model_\d+)").squeeze()
        ),
        "lgbm_transfer": lambda df: "LGBM Transfer (trained on "
        + df["data.subgroup"].str.split("/").str[-1]
        + ")",
        "lgbm_transfer_sample": lambda df: "LGBM Transfer (trained on "
        + df["data.subgroup"].str.split("/").str[-1]
        + " + "
        + df["data.add_from_other_subgroups_n"].astype(str)
        + ")",
        "llm_transfer": lambda df: "LLM Transfer (trained on "
        + df["data.subgroup"].str.split("/").str[-1]
        + ")",
        "base_mistral_oversampling": lambda df: "Fine-Tuned LLM "
        + df["model.name"]
        + " ("
        + df["data.oversampled_pos_fraction"].astype(str)
        + ")",
    }

    for group, name_func in group_to_name_mapping.items():
        if group in df["group"].unique():
            df.loc[df["group"] == group, "model_name"] = name_func(
                df[df["group"] == group]
            )

    # Map model_names to shorter names
    df["model_name_short"] = df["model_name"]
    for k, v in name_mapping.llm_short_names.items():
        df["model_name_short"] = df["model_name_short"].str.replace(k, v)
    for k, v in name_mapping.model_names_short.items():
        df["model_name_short"] = df["model_name_short"].str.replace(k, v)

    return df["model_name"], df["model_name_short"]


plot_df["model_name"], plot_df["model_name_short"] = get_model_name(plot_df)
if load_subgroups:
    plot_df_sg["model_name"], plot_df_sg["model_name_short"] = get_model_name(
        plot_df_sg
    )


def get_category(df):
    category_names = {
        "Risk Scores": ["risk_scores"],
        "ML Models": [
            "lgbm",
            "logreg_baseline",
            "cox_ph_baseline",
        ],
        "AdaCVD": [
            "different_base_llms",
            "base_llm_mistral",
            "base_llm_llama_8b",
            "base_llm_llama_3b",
            "base_llm_phi",
            "base_llm_gemma",
            # "mistral_separate_models",
        ],
        "Zero-Shot LLM": ["zeroshot", "zeroshot_mistral", "zeroshot_llama"],
    }

    df["category"] = None
    for category, groups in category_names.items():
        df.loc[df["group"].isin(groups), "category"] = category
    return df["category"]


plot_df["category"] = get_category(plot_df)
if load_subgroups:
    plot_df_sg["category"] = get_category(plot_df_sg)

evals_sizes = evals.groupby("directory").size()
full_eval_dirs = evals_sizes[evals_sizes == 81856].index.tolist()


# FEATURE GROUPS


def get_feature_group_names(df):
    df["data.feature_config.feature_groups"] = df[
        "data.feature_config.feature_groups"
    ].apply(
        lambda x: (
            sorted(
                x,
                key=lambda y: {
                    k: i for i, k in enumerate(name_mapping.feature_group_names.keys())
                }[
                    y
                ],  # TODO: name_mapping
            )
            if isinstance(x, list)
            else x
        )
    )

    feature_group_order = {
        k: i for i, k in enumerate(name_mapping.feature_group_names.keys())
    }

    df["feature_groups_long_name"] = df["data.feature_config.feature_groups"].apply(
        lambda x: (
            (" + ".join([name_mapping.feature_group_names[k]["long_name"] for k in x]))
            if isinstance(x, list)
            else x
        )
    )

    df["feature_groups_short_name"] = df["data.feature_config.feature_groups"].apply(
        lambda x: (
            " + ".join([name_mapping.feature_group_names[k]["short_name"] for k in x])
            if isinstance(x, list)
            else x
        )
    )

    # special cases:
    replace_dict = {
        "Fram + EB + Prevent": "Base",
        "Fram+EB+Prevent": "Base",
        "Framingham Risk Factors + Ethnic Background + Prevent Risk Factors": "Base",
    }

    for k, v in replace_dict.items():
        df["feature_groups_long_name"] = df["feature_groups_long_name"].str.replace(
            k, v
        )
        df["feature_groups_short_name"] = df["feature_groups_short_name"].str.replace(
            k, v
        )

    return df["feature_groups_long_name"], df["feature_groups_short_name"]


plot_df["feature_groups_long_name"], plot_df["feature_groups_short_name"] = (
    get_feature_group_names(plot_df)
)
if load_subgroups:
    plot_df_sg["feature_groups_long_name"], plot_df_sg["feature_groups_short_name"] = (
        get_feature_group_names(plot_df_sg)
    )


if load_subgroups:
    plot_df_sg.loc[plot_df_sg["group"] == "separate_models", "model_name_short"] = (
        "Fine-Tuned LLM ("
        + plot_df_sg.loc[
            plot_df_sg["group"] == "separate_models", "feature_groups_short_name"
        ]
        + ")"
    )


# %%
# ------------------------ VISUALS ------------------------ #

textwidth = 1.2 * 6.52  # inches

n = 12
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

sns.set_context(context, font_scale=0.9)
plt.rcParams["font.family"] = "Helvetica"  # "Arial"
# plt.rcParams["font.size"] = 4
plt.rcParams["xtick.bottom"] = True

plt.rcParams["axes.labelsize"] = plt.rcParams["xtick.labelsize"]
plt.rcParams["axes.titlesize"] = plt.rcParams["xtick.labelsize"]


# two color palette (True, False)
plt.colormaps.unregister("two_color_palette")
two_colors = [
    color_palette.BLUE[1],
    color_palette.BLUE[2],
]
cmap_two_colors = LinearSegmentedColormap.from_list("two_color_palette", two_colors)
plt.colormaps.register(cmap=cmap_two_colors)
cpal_two_color = sns.color_palette("two_color_palette", n_colors=2)

rotate_90 = True
ylabelpad = 15

# %%

relevant_metrics = {
    "roc_auc": "AUROC",
    "c_index": "C-Index",
    "average_precision_score": "Average Precision",
    "precision_score": "Precision",
    "recall_score": "Recall",
    "share_pos_pred": "Share Positive Preds",
    "share_pos": "Share Positive (True)",
    "f1": "F1",
    "balanced_acc": "Balanced Accuracy",
    "brier_score_loss": "Brier Score Loss",
}

higher_better = {
    "roc_auc": True,
    "c_index": True,
    "average_precision_score": True,
    "precision_score": True,
    "recall_score": True,
    "share_pos_pred": None,
    "share_pos": None,
    "f1": True,
    "balanced_acc": True,
    "brier_score_loss": False,
}

# %%

# ------------------------ BASE RISK FACTORS ------------------------ #

plot_df_base_risk_factors = plot_df[
    plot_df.apply(
        lambda x: (
            True
            if (
                x["data.feature_config.feature_groups"]
                in [
                    base_risk_factors,
                    # base_risk_factors_aha_aca,
                    # base_risk_factors_prevent,
                    # base_risk_factors_framingham,
                ]
                and x["group"]
                in [
                    "lgbm",
                    # "different_base_llms",
                    "logreg_baseline",
                    "cox_ph_baseline",
                    "base_llm_mistral",
                    "base_llm_llama_8b",
                    "base_llm_llama_3b",
                    "base_llm_phi",
                    "base_llm_gemma",
                    # "base_llm_mistral",k
                    # "zeroshot",
                    "zeroshot_mistral",
                    "zeroshot_llama",
                ]
            )
            or x["group"] == "risk_scores"
            else False
        ),
        axis=1,
    )
]

plot_df_base_risk_factors["median_roc_auc"] = plot_df_base_risk_factors.groupby(
    "directory"
)["roc_auc"].transform("median")
plot_df_base_risk_factors = plot_df_base_risk_factors.sort_values(
    by=["category", "median_roc_auc"], ascending=[False, True]
)
# plot_df_base_risk_factors = plot_df_base_risk_factors.sort_values(
#     by=["median_roc_auc"], ascending=[True]
# )

assert (
    plot_df_base_risk_factors.groupby("model_name")["directory"].nunique().max() <= 1
), "There are multiple directories for the same model_name."

category_color_dict = {
    "Zero-Shot LLM": color_palette.GREY[0],
    "Risk Scores": color_palette.GREY[1],
    "ML Models": color_palette.GREY[2],  # "#778da9", # #5d5d5d
    "AdaCVD": color_palette.BLUE[1],  # "#e63946",
}

# models to remove: "Fine-Tuned mistralai/Mistral-7B-Instruct-v0.2"
# plot_df_base_risk_factors = plot_df_base_risk_factors[
#     ~plot_df_base_risk_factors["model_name"].str.contains(
#         "mistralai/Mistral-7B-Instruct-v0.2"
#     )
# ]

for metric_col, metric_name in relevant_metrics.items():
    for mode in ["boxplot", "barplot", "pointplot"]:
        w = 0.55 * textwidth  # 0.5
        h = 0.65 * textwidth  # 0.5
        plt.figure(figsize=(w, h))
        remove_legend = True

        data = plot_df_base_risk_factors
        x = "model_name_short"
        y = metric_col
        hue = "category"

        kwargs = dict(palette=category_color_dict, hue_order=category_color_dict.keys())
        if mode == "boxplot":
            g = sns.boxplot(
                data=data,
                x=x,
                y=metric_col,
                hue=hue,
                fliersize=0,
                dodge=False,
                **kwargs,
            )
        elif mode == "barplot":
            g = sns.barplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                errorbar="pi",  # pi from bootstrapp is ci
                **kwargs,
            )
        elif mode == "pointplot":
            g = sns.pointplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                errorbar="pi",  # pi from bootstrapp is ci
                **kwargs,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if remove_legend:
            g.get_legend().remove()
        else:
            plt.legend(
                bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
                loc="lower left",
                # ncols=2,
                mode="expand",
                borderaxespad=0.0,
            )
        if rotate_90:
            plt.xticks(rotation=90, ha="center")
        else:
            plt.xticks(rotation=45, ha="right")

        ylabel = metric_name
        if higher_better[metric_col] is not None:
            if higher_better[metric_col]:
                ylabel = ylabel + r" $\longrightarrow$"
            else:
                ylabel = r"$\longleftarrow$ " + ylabel

        plt.ylabel(ylabel)

        plt.xlabel("Risk Prediction Model", labelpad=ylabelpad)
        plt.tight_layout()
        fig_name = f"base_risk_factors_{mode}_{metric_col}"
        plt.savefig(
            FIGURE_PATH / (fig_name + ".pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

        if remove_legend:
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
    break

# %% LLMs

plot_df_llms = plot_df[
    plot_df.apply(
        lambda x: (
            True
            if (
                x["data.feature_config.feature_groups"]
                in [
                    base_risk_factors,
                    # base_risk_factors_aha_aca,
                    # base_risk_factors_prevent,
                    # base_risk_factors_framingham,
                ]
                and x["group"]
                in [
                    # "different_base_llms",
                    "base_llm_mistral",
                    "base_llm_llama_8b",
                    "base_llm_llama_3b",
                    "base_llm_phi",
                    "base_llm_gemma",
                    "zeroshot_mistral",
                    "zeroshot_llama",
                ]
            )
            else False
        ),
        axis=1,
    )
]

plot_df_llms["median_roc_auc"] = plot_df_llms.groupby("directory")["roc_auc"].transform(
    "median"
)
plot_df_llms = plot_df_llms.sort_values(
    by=["category", "median_roc_auc"], ascending=[False, True]
)

assert (
    plot_df_llms.groupby("model_name")["directory"].nunique().max() <= 1
), "There are multiple directories for the same model_name."

plot_df_llms["category_2"] = plot_df_llms["category"].map(
    {"AdaCVD": "Fine-Tuned LLMs", "Zero-Shot LLM": "Zero-Shot LLMs"}
)

category_color_dict_2 = {
    "Zero-Shot LLMs": category_color_dict["Zero-Shot LLM"],
    "Fine-Tuned LLMs": category_color_dict["AdaCVD"],
}


for metric_col, metric_name in relevant_metrics.items():
    w = 0.45 * textwidth  # 0.5
    h = 0.55 * textwidth  # 0.5
    plt.figure(figsize=(w, h))
    remove_legend = True

    data = plot_df_llms
    x = "model_name_short"
    y = metric_col
    hue = "category_2"

    kwargs = dict(
        palette=category_color_dict_2, hue_order=["Zero-Shot LLMs", "Fine-Tuned LLMs"]
    )

    boxplot_args = dict(
        data=data,
        x=x,
        y=metric_col,
        hue=hue,
        fliersize=0,
        dodge=False,
        saturation=1.0,
        **kwargs,
    )

    g = sns.boxplot(
        **boxplot_args,
        linewidth=0,
        showfliers=False,
        boxprops=dict(alpha=0.5),
    )

    sns.boxplot(**boxplot_args, fill=False, legend=False)

    if remove_legend:
        g.get_legend().remove()
    else:
        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            # ncols=2,
            mode="expand",
            borderaxespad=0.0,
        )
    if rotate_90:
        plt.xticks(rotation=90, ha="center")
    else:
        plt.xticks(rotation=45, ha="right")

    ylabel = metric_name
    if higher_better[metric_col] is not None:
        if higher_better[metric_col]:
            ylabel = ylabel + r" $\longrightarrow$"
        else:
            ylabel = r"$\longleftarrow$ " + ylabel

    plt.ylabel(ylabel)

    plt.xlabel("Risk Prediction Model", labelpad=ylabelpad)
    plt.tight_layout()
    fig_name = f"llms_{metric_col}"
    plt.savefig(FIGURE_PATH / (fig_name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.show()

    if remove_legend:
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

        break

# %%

# ------------------------- CORRELATION -------------------------


mask = np.isin(evals["directory"], plot_df_base_risk_factors["directory"].unique())

evals_subset = evals[mask]

evals_subset["model_name_short"] = evals_subset["directory"].map(
    plot_df_base_risk_factors.set_index("directory")["model_name_short"].to_dict()
)
evals_subset["model_name"] = evals_subset["directory"].map(
    plot_df_base_risk_factors.set_index("directory")["model_name"].to_dict()
)
evals_subset = (
    evals_subset.reset_index()
    .drop_duplicates(subset=["model_name_short", "eid"])
    .set_index("eid")
)

evals_pivot = evals_subset.pivot(columns="model_name_short", values="y_pred_score")

# plot
correlation_matrix = evals_pivot.corr(method="kendall")
correlation_matrix.index = pd.CategoricalIndex(
    correlation_matrix.index,
    categories=plot_df_base_risk_factors["model_name_short"].unique(),
)
correlation_matrix.sort_index(level=0, inplace=True)
correlation_matrix = correlation_matrix[correlation_matrix.index]

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
correlation_matrix = correlation_matrix.mask(mask)

# diverging colormap
color_gradient = LinearSegmentedColormap.from_list(
    "custom_palette", [color_palette.BLUE[0], "#d8dbe2", color_palette.RED[1]]
)  # "#061A40", "#edede9",


w = 0.5 * textwidth
plt.figure(figsize=(w, w))
g = sns.heatmap(
    correlation_matrix,
    annot=True,
    annot_kws={
        "color": "white",
        "alpha": 1.0,
        "verticalalignment": "center",
        "horizontalalignment": "center",
        "size": 5,
    },
    fmt=".2f",
    cmap=color_gradient,  # "coolwarm",
    vmax=1,
    vmin=0,  # TODO: -1 or 0
    center=0.0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.3},
)
# remove grid
plt.grid(False)
g.set(xlabel=None, ylabel=None)
# plt.xlabel("Risk Prediction Model")
# plt.ylabel("Risk Prediction Model")

plt.savefig(
    FIGURE_PATH / "base_risk_factors_correlation_matrix.pdf",
    format="pdf",
    bbox_inches="tight",
)

# %%
# ------------------------- MORE RISK FACTOES -------------------------

plot_df_more_risk_factors = plot_df[(plot_df["group"] == "mistral_all_fgs")]

plot_df_more_risk_factors = plot_df_more_risk_factors[
    plot_df_more_risk_factors["feature_groups_short_name"].str.contains("\+")
    | (plot_df_more_risk_factors["feature_groups_short_name"] == "Base")
]

# exclude some feature groups
exclude = [
    "Fram",
    "Fram + EB",
    "Fram + Prevent",
    "Base + Sm + Sl + A",
    "Sm + Sl + A",
    "Base + Sm + Sl + PA + A",
    "Base + D",
    "Base + Sl",
    "Base + Sm",
    "Base + A",
    "Base + PA",
]
plot_df_more_risk_factors = plot_df_more_risk_factors[
    ~plot_df_more_risk_factors["feature_groups_short_name"].isin(exclude)
]

n_plus = plot_df_more_risk_factors["feature_groups_short_name"].str.count("\+")

plot_df_more_risk_factors["category"] = None
plot_df_more_risk_factors.loc[
    plot_df_more_risk_factors["feature_groups_short_name"] == "Base", "category"
] = "Base Risk Factors"

plot_df_more_risk_factors.loc[n_plus == 1, "category"] = "Base + Single Group"

# plot_df_more_risk_factors.loc[n_plus > 1, "category"] = "Base + Combinations"


plot_df_more_risk_factors["feature_groups_display_name"] = None
plot_df_more_risk_factors.loc[
    plot_df_more_risk_factors["category"] == "Base Risk Factors",
    "feature_groups_display_name",
] = "Base"

plot_df_more_risk_factors.loc[n_plus == 1, "feature_groups_display_name"] = (
    plot_df_more_risk_factors["feature_groups_short_name"]
)

plot_df_more_risk_factors.loc[n_plus > 1, "feature_groups_display_name"] = (
    plot_df_more_risk_factors["feature_groups_short_name"]
)

plot_df_more_risk_factors.loc[n_plus == 9, "feature_groups_display_name"] = (
    "All"  # "All Patient Information"
)


plot_df_more_risk_factors["feature_groups_display_name_long"] = None
plot_df_more_risk_factors.loc[
    plot_df_more_risk_factors["category"] == "Base Risk Factors",
    "feature_groups_display_name_long",
] = "Base Risk Factors"

plot_df_more_risk_factors.loc[n_plus == 1, "feature_groups_display_name_long"] = (
    plot_df_more_risk_factors["feature_groups_long_name"]
)

plot_df_more_risk_factors.loc[n_plus > 1, "feature_groups_display_name_long"] = (
    plot_df_more_risk_factors["feature_groups_long_name"]
)

plot_df_more_risk_factors.loc[n_plus == 9, "feature_groups_display_name_long"] = (
    "All Patient Information"
)


# remove combinations except for 9
plot_df_more_risk_factors = plot_df_more_risk_factors[~((n_plus > 1) & (n_plus != 9))]
plot_df_more_risk_factors.loc[n_plus > 1, "category"] = "All Patient Information"

plot_df_more_risk_factors = pd.concat(
    [plot_df_more_risk_factors, plot_df_base_risk_factors]
)

color_dict = {
    **category_color_dict,
    "Base Risk Factors": category_color_dict["AdaCVD"],
    "Base + Single Group": color_palette.BLUE[2],  # "#00B4D8",
    # "Base + Combinations": "#1B3B6F",
    "All Patient Information": color_palette.TEAL[0],  # "#1B3B6F",  # "blue"
}

# remove certain models
to_remove = [
    "Zero-Shot LLM Llama-3B",
    "Fine-Tuned LLM Phi-mini-3B",
    "Fine-Tuned LLM Gemma-2B",
    "Fine-Tuned LLM Llama-3B",
    "Fine-Tuned LLM Llama-8B",
]

plot_df_more_risk_factors = plot_df_more_risk_factors[
    ~plot_df_more_risk_factors["model_name_short"].isin(to_remove)
]

# replace name:
replace_names = {
    "Fine-Tuned LLM Mistral-7B": "AdaCVD",
    "Zero-Shot LLM Mistral-7B": "Zero-Shot LLM",
}
for k, v in replace_names.items():
    plot_df_more_risk_factors["model_name_short"] = plot_df_more_risk_factors[
        "model_name_short"
    ].str.replace(k, v)
    plot_df_more_risk_factors["model_name"] = plot_df_more_risk_factors[
        "model_name"
    ].str.replace(k, v)


plot_df_more_risk_factors["category"] = pd.Categorical(
    plot_df_more_risk_factors["category"],
    categories=list(color_dict.keys()),
    ordered=True,
)

plot_df_more_risk_factors["mean_roc_auc"] = plot_df_more_risk_factors.groupby(
    ["directory"]
)["roc_auc"].transform("mean")


plot_df_more_risk_factors["median_roc_auc"] = plot_df_more_risk_factors.groupby(
    "directory"
)["roc_auc"].transform("median")
plot_df_more_risk_factors = plot_df_more_risk_factors.sort_values(
    by=["category", "median_roc_auc"], ascending=[True, True]
)

plot_df_more_risk_factors["display_name"] = None

plot_df_more_risk_factors.loc[
    plot_df_more_risk_factors["category"].isin(
        ["Zero-Shot LLM", "Risk Scores", "ML Models", "AdaCVD"]
    ),
    "display_name",
] = plot_df_more_risk_factors["model_name_short"]

plot_df_more_risk_factors.loc[
    plot_df_more_risk_factors["category"].isin(
        [
            "Base Risk Factors",
            "Base + Single Group",
            # "Base + Combinations",
            "All Patient Information",
        ]
    ),
    "display_name",
] = plot_df_more_risk_factors["feature_groups_display_name"]

x_order = plot_df_more_risk_factors["display_name"].unique()

for metric_col, metric_name in relevant_metrics.items():
    for mode in ["boxplot", "barplot", "pointplot"]:
        w = 1.1 * textwidth
        h = 0.35 * w
        plt.figure(figsize=(w, h))
        remove_legend = True

        data = plot_df_more_risk_factors
        x = "display_name"
        y = metric_col
        hue = "category"

        kwargs = dict(palette=color_dict)

        if mode == "boxplot":
            boxplot_args = dict(
                data=data,
                x=x,
                y=metric_col,
                hue=hue,
                fliersize=0.1,
                dodge=False,
                # linecolor="grey",
                # fill=False,
                saturation=1.0,
                **kwargs,
            )

            g = sns.boxplot(
                **boxplot_args,
                linewidth=0,
                showfliers=False,
                boxprops=dict(alpha=0.5),
            )

            sns.boxplot(**boxplot_args, fill=False, legend=False)

            # horizontal line
            most_comp_rs_median = plot_df_more_risk_factors[
                plot_df_more_risk_factors[x] == "PREVENT Risk Score"
            ][metric_col].median()

            best_median = plot_df_more_risk_factors[
                plot_df_more_risk_factors[x] == "All Patient Information"
            ][metric_col].median()

            print(f"Most competitive risk score median: {most_comp_rs_median}")
            print(f"Best median: {best_median}")
            print(
                f"Relative improvement: {100 * (best_median - most_comp_rs_median) / most_comp_rs_median:.2f}%"
            )
            add_annotation = False
            if add_annotation:
                plt.axhline(
                    y=most_comp_rs_median,
                    xmin=1 - 1 / len(data[x].unique()),
                    xmax=1.0,
                    color=color_dict["Risk Scores"],
                    linestyle="--",
                    linewidth=1,
                )

                # annotate with relative improvement
                g.axes.annotate(
                    f"{100 * (best_median - most_comp_rs_median) / most_comp_rs_median:.1f}%",
                    xy=(1.0, best_median),
                    xytext=(1.0, best_median),
                    textcoords="axes fraction",
                    ha="left",
                    va="center",
                )

        elif mode == "barplot":
            g = sns.barplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                errorbar="pi",  # pi from bootstrapp is ci
                **kwargs,
            )
        elif mode == "pointplot":
            g = sns.pointplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                errorbar="pi",  # pi from bootstrapp is ci
                **kwargs,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if remove_legend:
            g.get_legend().remove()
        else:
            plt.legend(
                bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
                loc="lower left",
                ncols=1,
                mode="expand",
                borderaxespad=0.0,
            )
        # plt.yticks(va="left")
        rotate_90 = False
        if rotate_90:
            plt.xticks(rotation=90, ha="center")
        else:
            plt.xticks(rotation=60, ha="right")

        ylabel = metric_name
        if higher_better[metric_col] is not None:
            if higher_better[metric_col]:
                ylabel = ylabel + r" $\longrightarrow$"
            else:
                ylabel = r"$\longleftarrow$ " + ylabel

        plt.ylabel(ylabel)
        plt.xlabel("")

        # if metric_col == "roc_auc":
        #     plt.ylim(0.59, 0.79)

        # plt.tight_layout()
        fig_name = f"more_risk_factors_{mode}_{metric_col}"
        plt.savefig(
            FIGURE_PATH / (fig_name + ".pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

        if remove_legend:
            # plot again to get alpha=1.0

            plt.figure(figsize=(w, h))
            g = sns.boxplot(
                **boxplot_args,
                linewidth=0,
                showfliers=False,
            )
            plt.close()

            fig_legend = plt.figure()
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
        break
    # break


# %%
# ------------------------ SCORE DISTRIBUTION & CALIBRATION ------------------------ #
# calibration & distribution, selected models


two_colors = [
    color_palette.BLUE[0],
    color_palette.TEAL[0],
]


evals_plot = evals[
    evals["directory"].isin(plot_df_more_risk_factors["directory"].unique())
    | evals["directory"].isin(plot_df_base_risk_factors["directory"].unique())
]

directory_to_model_name_short = {
    **plot_df_base_risk_factors.set_index("directory")["model_name_short"].to_dict(),
    **plot_df_more_risk_factors.set_index("directory")["model_name_short"].to_dict(),
}

evals_plot["model_name_short"] = evals_plot["directory"].map(
    directory_to_model_name_short
)

evals_plot["display_name"] = evals_plot["directory"].map(
    plot_df_more_risk_factors.set_index("directory")["display_name"].to_dict(),
)

evals_plot["feature_groups_display_name_long"] = evals_plot["directory"].map(
    plot_df_more_risk_factors.set_index("directory")[
        "feature_groups_display_name_long"
    ].to_dict(),
)

evals_plot["model_and_features"] = evals_plot["model_name_short"].copy()
evals_plot.loc[evals_plot["group"] == "mistral_all_fgs", "model_and_features"] = (
    # evals_plot.loc[evals_plot["group"] == "mistral_all_fgs", "model_name_short"]
    "AdaCVD ("
    # + " – "
    + evals_plot.loc[
        evals_plot["group"] == "mistral_all_fgs", "feature_groups_display_name_long"
    ]
    + ")"
)
evals_plot["y_true"] = evals_plot["y_true"].astype(bool)

# order like this: Risk Scores < ML Models < Zero-Shot LLM < AdaCVD
plot_df_more_risk_factors["category_2"] = pd.Categorical(
    plot_df_more_risk_factors["category"],
    categories=[
        "Risk Scores",
        "ML Models",
        "Zero-Shot LLM",
        "AdaCVD",
        "Base Risk Factors",
        "Base + Single Group",
        "Base + Combinations",
        "All Patient Information",
    ],
    ordered=True,
)

model_name_order = plot_df_more_risk_factors.sort_values(
    by=["category_2", "median_roc_auc"]
)["model_name_short"].unique()

evals_plot["model_name_order"] = evals_plot["model_name_short"].map(
    {v: i for i, v in enumerate(model_name_order)}
)

evals_plot = evals_plot.sort_values(by=["model_name_order"])
col_order = evals_plot["model_and_features"].unique()

only_few_models = True
if only_few_models:
    # subset only
    models = [
        "Zero-Shot LLM",
        "ACC/AHA",
        "QRISK",
        "SCORE",
        "Framingham",
        "PREVENT",
        "LogReg",
        "Cox PH",
        "LGBM",
        "AdaCVD (Base Risk Factors)",
        "AdaCVD (All Patient Information)",
    ]
    evals_plot = evals_plot[evals_plot["model_and_features"].isin(models)]
    col_order = models

# h = 0.273*textwidth / 2
# w = 0.5*textwidth

evals_plot["CVD_outcome"] = evals_plot["y_true"].map({True: "CVD", False: "No CVD"})


g = sns.FacetGrid(
    evals_plot,
    col="model_and_features",
    col_wrap=2 if only_few_models else 3,
    height=0.18 * textwidth,
    aspect=1.6,
    sharex=True,
    sharey=True,  # False,
    hue="CVD_outcome",
    palette=two_colors,
    col_order=col_order,
    hue_order=["No CVD", "CVD"],
    # gridspec_kws={"wspace": 0.0001, "hspace": 0.0001},
)

g.map(
    sns.kdeplot,
    "y_pred_score",
    # bins=30,
    # kde=True,
    fill=True,
    # stat="density",
    common_norm=False,
    hue_order=[True, False],
    alpha=0.5,
    bw_adjust=1.2,
)
g.set_titles("{col_name}", size=8)

g.add_legend(
    title="Observed Outcome",
    ncol=2,
    labelspacing=0.5,
    columnspacing=1.0,
    loc="upper center",
    bbox_to_anchor=(0.0, 1.05, 1.0, 1.05),
)
plt.setp(g.legend.get_title(), fontsize="8")


# plt.tight_layout()
g.set_axis_labels("Risk Prediction", "Density")

# remove grid


# g.fig.suptitle("Distribution of Risk Predictions", y=1.01)

if only_few_models:
    for i, ax in enumerate(g.axes.flat):
        ax.set_xlim(-0.075, 0.8)
        # if i == 0:
        #     ax.set_ylim(0, 12)
        # if i > 0:
        #     ax.set_ylim(0, 6)
    # ax.grid(False)

# plt.tight_layout()
plt.savefig(
    FIGURE_PATH / f"score_distribution{'_few' if only_few_models else ''}.pdf",
    format="pdf",
    bbox_inches="tight",
)

# %%
# calibration plot
bins = np.linspace(0, 1, 20)
evals_plot["bin"] = pd.cut(evals_plot["y_pred_score"], bins=bins)
calibration = evals_plot.groupby(["bin", "model_and_features"])[
    ["y_pred_score", "y_true"]
].mean()

calibration[evals_plot.groupby(["bin", "model_and_features"]).size() < 20] = np.nan


# Define figure size
# w = 0.4 * textwidth
# h = w

w = 0.4 * textwidth
h = 0.6 * w

palette = {
    "Framingham": color_dict["Risk Scores"],
    "PREVENT": color_dict["Risk Scores"],
    "LGBM": color_dict["ML Models"],
    "Zero-Shot LLM": color_dict["Zero-Shot LLM"],
    "AdaCVD (Base Risk Factors)": color_dict["Base Risk Factors"],
    "AdaCVD (All Patient Information)": color_dict["All Patient Information"],
}

plt.figure(figsize=(w, h))
g = sns.lineplot(
    data=calibration,
    x="y_pred_score",
    y="y_true",
    hue="model_and_features",
    style="model_and_features",
    markers=True,
    hue_order=models if only_few_models else col_order,
    alpha=0.8,
    palette=palette,
    # linewidth=1.5,
    # markersize=5,
)

# plt.title("Calibration Plot")
plt.xlabel("Predicted Risk")
plt.ylabel("Observed Risk")

# Reference diagonal line
max_val = 0.8
plt.plot([0, max_val], [0, max_val], color="black", linestyle="--", alpha=0.5)

remove_legend = True
if remove_legend:
    g.get_legend().remove()
else:
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncols=2,
        mode="expand",
        borderaxespad=0.0,
    )

# Save the main plot
fig_name = f"calibration_plot{'_few' if only_few_models else ''}"
if save:
    plt.savefig(
        FIGURE_PATH / f"{fig_name}.pdf",
        bbox_inches="tight",
        format="pdf",
    )
plt.show()

if remove_legend:
    fig_legend = plt.figure(figsize=(w, h))
    ax = fig_legend.gca()
    legend_handles, legend_labels = g.get_legend_handles_labels()
    # legend_labels = [x.replace(" – ", "\n") for x in legend_labels]
    ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        frameon=False,
        ncol=1,
        labelspacing=0.5,
        columnspacing=1.0,
    )
    ax.axis("off")
    fig_legend.savefig(
        FIGURE_PATH / f"{fig_name}_legend.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    fig_legend.show()

# SURVIVAL CURVE
# %%
evals_plot["days_since_baseline"] = evals_plot[
    "MACE_ADO_EXTENDED_days_from_baseline"
].fillna(365 * 10 + 100)

quantiles = [0, 0.95, 0.99, 1.0]
labels = [
    f"Lowest {int(quantiles[1] * 100)}%",
    f"{int(quantiles[1] * 100)}% - {int(quantiles[2] * 100)}%",
    f"Highest {int(100-quantiles[2] * 100)}%",
]

risk_color_mapping = {
    labels[0]: color_palette.BLUE[1],
    labels[1]: color_palette.BLUE[3],
    # labels[2]: color_palette.TEAL[0],
    labels[2]: color_palette.RED[1],
}

evals_plot["Risk Quantile"] = evals_plot.groupby("directory")["y_pred_score"].transform(
    lambda x: pd.qcut(x.rank(method="first"), q=quantiles, labels=labels)
)

evals_plot["y_true"] = evals_plot["y_true"].astype(bool)


g = sns.FacetGrid(
    evals_plot,
    col="model_and_features",
    col_wrap=4 if only_few_models else 3,
    height=0.18 * textwidth,
    aspect=1.6,
    col_order=col_order,
    palette=risk_color_mapping,
    hue="Risk Quantile",
    sharex=True,
    sharey=True,
)

g.map(
    sns.ecdfplot,
    "days_since_baseline",
    stat="percent",
    # complementary=True,
    palette=risk_color_mapping,
    # linewidth=2,
)

g.set(ylim=(0, 50), xlim=(0, 365 * 10))
g.set_titles("{col_name}", size=8)

g.set_axis_labels("Days since Baseline Assessment", "Percent")

# g.add_legend(
#     title=None,  # "Risk Quantile",
#     ncol=3,
#     labelspacing=0.5,
#     columnspacing=1.0,
#     loc="upper center",
#     bbox_to_anchor=(0.0, 1.05, 1.0, 1.05),
# )

# remove grid


# g.fig.suptitle("Distribution of Risk Predictions", y=1.01)

# ax.grid(False)

plt.savefig(
    FIGURE_PATH / f"survival_curves{'_few' if only_few_models else ''}.pdf",
    format="pdf",
    bbox_inches="tight",
)


# %%
# ------------------------ SEPARATE/FLEXIBLE MODELS COMPARISON ------------------------ #

plot_df_more_risk_factors_comp = plot_df[
    # (plot_df["group"] == "mistral_separate_models")
    (plot_df["group"] == "mistral_all_fgs")
    | (plot_df["group"] == "mistral_full_model")
    | (plot_df["group"] == "mistral_flexible_model")
    # | (plot_df["group"] == "oversampling")
    # | (plot_df["group"] == "base_llm_mistral")
    # | (plot_df["group"] == "lgbm_all_fgs_params")
    | (plot_df["group"] == "lgbm_all_fgs_rounded")
    # | (plot_df["group"] == "cox_ph_baseline")
    # | (plot_df["group"] == "lgbm")
    # above
    # | (plot_df["group"] == "full_model")
    # | (plot_df["group"] == "flexible_sampling_improved")
    # (plot_df["group"] == "separate_models")
    # | (plot_df["group"] == "separate_models_2")
    # | (plot_df["group"] == "separate_models_extended")
    # | (plot_df["group"] == "separate_models_same_subset")
    # | (plot_df["group"] == "separate_models_MH")
    # | (plot_df["group"] == "separate_models_MH_2")
    # (plot_df["group"] == "base_risk_factors")
    | (plot_df["group"] == "lgbm_flexible_model")
    # (plot_df["group"] == "lgbm_all_combs")
]

# plot_df_more_risk_factors_comp = plot_df_more_risk_factors_comp[
#     plot_df_more_risk_factors_comp["directory"].isin(full_eval_dirs)
# ]

plot_df_more_risk_factors_comp = plot_df_more_risk_factors_comp[
    (
        plot_df_more_risk_factors_comp["feature_groups_short_name"].str.contains("\+")
        | (plot_df_more_risk_factors_comp["feature_groups_short_name"] == "Base")
    )
    #     & (
    #         ~plot_df_more_risk_factors_comp["feature_groups_short_name"].isin(
    #             ["Sm + Sl + A", "Fram + EB", "Fram + Prevent"]
    #         )
    #     )
    # & (plot_df_more_risk_factors_comp["feature_groups_short_name"].str.count("\+") > 1)
]

fgs = plot_df_more_risk_factors_comp.groupby("feature_groups_short_name")[
    "group"
].nunique()
fgs = fgs[fgs >= 1].index.values

mask = plot_df_more_risk_factors_comp["feature_groups_short_name"].isin(fgs)

plot_df_more_risk_factors_comp = plot_df_more_risk_factors_comp[mask]

# plot_df_more_risk_factors_comp["dir_base"] = plot_df_more_risk_factors_comp[
#     "directory"
# ].str.extract(r"2025_02_01_report/(.+?)/\d{3}/\d{3}/full")


plot_df_more_risk_factors_comp["mean_roc_auc"] = plot_df_more_risk_factors_comp.groupby(
    ["directory"]
)["roc_auc"].transform("median")

plot_df_more_risk_factors_comp = plot_df_more_risk_factors_comp.sort_values(
    by=["mean_roc_auc"], ascending=[True]
)

# # map feature_groups_short_name to display_name
# plot_df_more_risk_factors_comp["display_name"] = (
#     plot_df_more_risk_factors_comp["feature_groups_short_name"].map(
#         plot_df_more_risk_factors.drop_duplicates(subset="feature_groups_short_name").set_index("feature_groups_short_name")["display_name"]
#     )
# )

n_plus = plot_df_more_risk_factors_comp["feature_groups_short_name"].str.count("\+")

plot_df_more_risk_factors_comp["category"] = None
plot_df_more_risk_factors_comp.loc[
    plot_df_more_risk_factors_comp["feature_groups_short_name"] == "Base", "category"
] = "Base Risk Factors"

plot_df_more_risk_factors_comp.loc[n_plus == 1, "category"] = "Base + Single Group"

plot_df_more_risk_factors_comp.loc[n_plus > 1, "category"] = "Base + Combinations"


plot_df_more_risk_factors_comp["feature_groups_display_name"] = None
plot_df_more_risk_factors_comp.loc[
    plot_df_more_risk_factors_comp["category"] == "Base Risk Factors",
    "feature_groups_display_name",
] = "Base"

plot_df_more_risk_factors_comp.loc[n_plus == 1, "feature_groups_display_name"] = (
    plot_df_more_risk_factors_comp["feature_groups_short_name"]
)

plot_df_more_risk_factors_comp.loc[n_plus > 1, "feature_groups_display_name"] = (
    plot_df_more_risk_factors_comp["feature_groups_short_name"]
)

plot_df_more_risk_factors_comp.loc[n_plus == 9, "feature_groups_display_name"] = "All"

# only base, base + X, and full
plot_df_more_risk_factors_comp = plot_df_more_risk_factors_comp[
    (n_plus == 9) | (n_plus <= 1)
]


plot_df_more_risk_factors_comp["display_name"] = None

plot_df_more_risk_factors_comp.loc[
    plot_df_more_risk_factors_comp["category"].isin(
        ["Zero-Shot LLM", "Risk Scores", "ML Models", "AdaCVD"]
    ),
    "display_name",
] = plot_df_more_risk_factors_comp["model_name_short"]

plot_df_more_risk_factors_comp.loc[
    plot_df_more_risk_factors_comp["category"].isin(
        [
            "Base Risk Factors",
            "Base + Single Group",
            "Base + Combinations",
            "All Patient Information",
        ]
    ),
    "display_name",
] = plot_df_more_risk_factors_comp["feature_groups_display_name"]

plot_df_more_risk_factors_comp["feature_groups_short_name"] = (
    plot_df_more_risk_factors_comp["feature_groups_short_name"].replace(
        "Base + ICD + PM + LE + FH + SD + PRS + MH + BS + UA", "All"
    )
)

order = [
    x for x in x_order if x in plot_df_more_risk_factors_comp["display_name"].unique()
]

# order in "feature_groups_short_name"

display_name_to_short_name = plot_df_more_risk_factors_comp.set_index("display_name")[
    "feature_groups_short_name"
].to_dict()
order = [display_name_to_short_name[x] for x in order]


# model_name map

group_model_name_map = {
    "mistral_all_fgs": "AdaCVD (Feature Expert Models)",
    "lgbm_all_fgs_params": "LGBM (Feature Expert Models)",
    "lgbm_all_fgs_rounded": "LGBM (Feature Expert Models)",
    "mistral_flexible_model": "AdaCVD-Flex (Single Model)",
    "lgbm_flexible_model": "LGBM (All Patient Info; Single Model)",
    "mistral_full_model": "AdaCVD (All Patient Info; Single Model)",
}

plot_df_more_risk_factors_comp["model_name_short"] = plot_df_more_risk_factors_comp[
    "group"
].map(group_model_name_map)

for metric_col, metric_name in relevant_metrics.items():
    w = 1.1 * textwidth
    h = 0.3 * w
    plt.figure(figsize=(w, h))
    remove_legend = True

    custom_palette = {
        "AdaCVD (Feature Expert Models)": color_palette.GREY[1],  # "blue"
        "AdaCVD-Flex (Single Model)": color_palette.TEAL[0],
        "AdaCVD (All Patient Info; Single Model)": color_palette.BLUE[1],  # "#415a77",
        # # "LGBM: Separate Models": "#A0A0A0",
        # "LGBM (All Patient Info; Single Model)": "#667798",  # "#5d5d5d",
    }

    custom_linestyle = {
        "AdaCVD (Feature Expert Models)": "dotted",
        "AdaCVD-Flex (Single Model)": "solid",
        "AdaCVD (All Patient Info; Single Model)": "solid",
        # # "LGBM: Separate Models": "solid",
        # "LGBM (All Patient Info; Single Model)": "solid",
    }

    # g = sns.pointplot(
    #     data=plot_df_more_risk_factors_comp,
    #     x="feature_groups_short_name",  # "display_name",
    #     y=metric_col,
    #     hue="model_name_short",
    #     order=order,
    #     palette=custom_palette,
    #     hue_order=custom_palette.keys(),
    #     # alpha=0.8,
    #     errorbar="pi",
    #     # linestyles=[*custom_linestyle.values()],
    #     markers=["o", "o", "o", "x", "x"],
    #     join=False,
    #     dodge=True,
    # )
    as_boxplot = True
    if as_boxplot:
        boxplot_args = dict(
            data=plot_df_more_risk_factors_comp,
            x="feature_groups_short_name",
            y=metric_col,
            hue="model_name_short",
            fliersize=0.1,
            dodge=True,
            gap=0.15,
            saturation=1.0,
            # order=order,
            palette=custom_palette,
            hue_order=custom_palette.keys(),
        )

        g = sns.boxplot(
            **boxplot_args,
            linewidth=0,
            showfliers=False,
            boxprops=dict(alpha=0.5),
            order=order,
        )

        sns.boxplot(**boxplot_args, fill=False, legend=False, order=order)

    else:
        g = sns.pointplot(
            data=plot_df_more_risk_factors_comp,
            x="feature_groups_short_name",  # "display_name",
            y=metric_col,
            hue="model_name_short",
            order=order,
            palette=custom_palette,
            hue_order=custom_palette.keys(),
            # alpha=0.8,
            errorbar="pi",
            linestyles=[*custom_linestyle.values()],
            # markers=["o", "o", "o", "x", "x"],
            # join=False,
        )
    for child in ax.get_children():
        if hasattr(child, "set_alpha"):
            child.set_alpha(0.3)

    plt.xticks(rotation=90, ha="center")
    plt.grid(True)

    if remove_legend:
        g.get_legend().remove()
    else:
        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=1,
            mode="expand",
            borderaxespad=0.0,
        )

    if rotate_90:
        plt.xticks(rotation=90, ha="center")
    else:
        plt.xticks(rotation=60, ha="right")

    ylabel = metric_name
    if higher_better[metric_col] is not None:
        if higher_better[metric_col]:
            ylabel = ylabel + r" $\longrightarrow$"
        else:
            ylabel = r"$\longleftarrow$ " + ylabel

    plt.ylabel(ylabel)

    plt.xlabel("")

    # plt.tight_layout()
    fig_name = f"comparison_separate_flexible_{metric_col}"
    plt.savefig(FIGURE_PATH / (fig_name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.show()

    if remove_legend:
        fig_legend = plt.figure()
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
            FIGURE_PATH / (fig_name + "_legend.pdf"), format="pdf", bbox_inches="tight"
        )
        fig_legend.show()

    break


# %%

plot_df_flexible_model = plot_df[plot_df["group"] == "base_mistral_oversampling"]


plot_df_flexible_model["mean_roc_auc"] = plot_df_flexible_model.groupby(["directory"])[
    "roc_auc"
].transform("mean")

plot_df_flexible_model = plot_df_flexible_model.sort_values(
    by=["mean_roc_auc"], ascending=[True]
)

for metric_col, metric_name in relevant_metrics.items():
    plt.figure(figsize=(6, 4.5))
    remove_legend = True

    g = sns.boxplot(
        plot_df_flexible_model,
        x="data.oversampled_pos_fraction",
        y=metric_col,
        hue="feature_groups_long_name",
        fliersize=0,
    )  # , errorbar="pi")

    if remove_legend:
        g.get_legend().remove()
    else:
        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            # ncols=2,
            mode="expand",
            borderaxespad=0.0,
        )
    if rotate_90:
        plt.xticks(rotation=90, ha="center")
    else:
        plt.xticks(rotation=45, ha="right")

    ylabel = metric_name
    if higher_better[metric_col] is not None:
        if higher_better[metric_col]:
            ylabel = ylabel + r" $\longrightarrow$"
        else:
            ylabel = r"$\longleftarrow$ " + ylabel

    plt.ylabel(ylabel)

    plt.xlabel("Patient Information")
    # plt.tight_layout()
    fig_name = f"oversampling_{metric_col}"
    plt.savefig(FIGURE_PATH / (fig_name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.show()

    if remove_legend:
        fig_legend = plt.figure()
        ax = fig_legend.gca()
        ax.legend(*g.get_legend_handles_labels(), loc="center", frameon=False)
        ax.axis("off")
        fig_legend.savefig(
            FIGURE_PATH / (fig_name + "_legend.pdf"), format="pdf", bbox_inches="tight"
        )
        fig_legend.show()


# %%

# precision recall curve

for oversampling in [0.0, 0.2]:

    evals_subset = evals[
        evals["directory"].isin(
            plot_df_flexible_model[
                plot_df_flexible_model["data.oversampled_pos_fraction"] == oversampling
            ].directory.unique()
        )
    ]

    precision, recall, thresholds = precision_recall_curve(
        y_true=evals_subset["y_true"], y_score=evals_subset["y_pred_score"]
    )

    plt.figure(figsize=(6, 4.5))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlim(0, 1)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Oversampling: {oversampling}")
    plt.legend()


# %%
# ------------------------ DATA EFFICIENCY ------------------------ #
plot_df_data_efficiency = plot_df[
    plot_df["group"].isin(
        [
            "data_efficiency_lgbm",
            "data_efficiency_mistral",
            "base_llm_mistral",
            "cox_data_efficiency",
        ]
    )
]

plot_df_data_efficiency["mean_roc_auc"] = plot_df_data_efficiency.groupby(
    ["directory"]
)["roc_auc"].transform("mean")

plot_df_data_efficiency = plot_df_data_efficiency.sort_values(
    by=["mean_roc_auc"], ascending=[True]
)

n_all = 400_000  # TODO: fill with correct number
plot_df_data_efficiency.loc[
    plot_df_data_efficiency["directory"].str.contains("all"),
    "data.num_training_samples",
] = n_all

plot_df_data_efficiency.loc[
    plot_df_data_efficiency["group"] == "base_llm_mistral",
    "data.num_training_samples",
] = n_all


plot_df_data_efficiency = plot_df_data_efficiency[
    plot_df_data_efficiency["training.random_seed"] == 0
]

for metric_col, metric_name in relevant_metrics.items():
    plt.figure(figsize=(6, 4.5))
    remove_legend = False

    g = sns.lineplot(
        data=plot_df_data_efficiency,
        x="data.num_training_samples",
        hue="model_name",
        # style="training.random_seed",
        y=metric_col,
        markersize=8,
        marker="o",
    )

    if remove_legend:
        g.get_legend().remove()
    else:
        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            # ncols=2,
            mode="expand",
            borderaxespad=0.0,
        )
    if rotate_90:
        plt.xticks(rotation=90, ha="center")
    else:
        plt.xticks(rotation=45, ha="right")

    ylabel = metric_name
    if higher_better[metric_col] is not None:
        if higher_better[metric_col]:
            ylabel = ylabel + r" $\longrightarrow$"
        else:
            ylabel = r"$\longleftarrow$ " + ylabel

    plt.ylabel(ylabel)

    plt.xlabel("Number of Datapoints used for Training")
    plt.xscale("log")
    # plt.tight_layout()
    fig_name = f"data_efficiency_{metric_col}"
    plt.savefig(FIGURE_PATH / (fig_name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.show()

    if remove_legend:
        fig_legend = plt.figure()
        ax = fig_legend.gca()
        ax.legend(*g.get_legend_handles_labels(), loc="center", frameon=False)
        ax.axis("off")
        fig_legend.savefig(
            FIGURE_PATH / (fig_name + "_legend.pdf"), format="pdf", bbox_inches="tight"
        )
        fig_legend.show()

    break

    # plt.ylim(0.6, 0.8)
    # plt.xlim(20, 400_000)
    # g.set(xscale="log")
    # g.set_xticks(np.log(plot_df["evaluation_step_n"].sort_values().unique()))
    # g.set_xticklabels(plot_df["evaluation_step_n"].sort_values().unique())

# %%
# ------------------------ FLEXIBLE SAMPLING ------------------------ #
# compare individual models with flexible models

cpal = sns.color_palette("custom_palette", n_colors=5)
sns.set_palette(palette=cpal)

plot_df_separate_flexible = plot_df[
    (plot_df["group"] == "separate_models")
    | (plot_df["group"] == "flexible_model")
    # | (plot_df["group"] == "flexible_model_new")
    # | (plot_df["group"] == "flexible_sampling_new_combs")
    # | (plot_df["group"] == "flexible_sampling_new_models")
    | (plot_df["group"] == "flexible_sampling_improved")
    | (plot_df["group"] == "flexible_sampling_improved_more_combs")
    | (plot_df["group"] == "lgmb_baseline_more_features")
    | (plot_df["group"] == "lgbm_all_combs")
]

plot_df_separate_flexible = plot_df_separate_flexible[
    plot_df_separate_flexible["directory"].isin(full_eval_dirs)
]

plot_df_separate_flexible = plot_df_separate_flexible[
    (
        plot_df_separate_flexible["feature_groups_short_name"].str.contains("\+")
        | (plot_df_separate_flexible["feature_groups_short_name"] == "Base")
    )
    & (
        ~plot_df_separate_flexible["feature_groups_short_name"].isin(
            [
                "Sm + Sl + A",
                "Fram + EB",
                "Fram + Prevent",
                "Base + Sl",
                "Base + A",
                "Base + Sm",
                "Base + D",
                "Base + Sm + Sl + A",
                "Base + Sm + Sl + PA + A",
                "Base + LE + FH",
                "Base + ICD + MH",
            ]
        )
    )
]


plot_df_separate_flexible["order"] = plot_df_separate_flexible.groupby(
    "feature_groups_short_name"
)["roc_auc"].transform("mean")

plot_df_separate_flexible = plot_df_separate_flexible.sort_values(
    by=["order"], ascending=[True]
)

order = (
    plot_df_separate_flexible[plot_df_separate_flexible["group"] == "separate_models"]
    .groupby("feature_groups_short_name")["roc_auc"]
    .mean()
    .sort_values()
    .index.values
)

hue_name = {
    "Separate Models New": "Separate Models",
    "Improved Sampling model_031": "Flexible Model I",
    # "Improved Sampling model_000": "Flexible Model II",
    "model_000": "Flexible Model II",
    "model_002": "Flexible Model III",
    "Gradient Boosted Trees": "Gradient Boosted Trees",
}

plot_df_separate_flexible["model_name_hue"] = plot_df_separate_flexible[
    "model_name"
].map(hue_name)

remove_legend = False
for metric_col, metric_name in relevant_metrics.items():
    plt.figure(figsize=(10, 6))

    g = sns.pointplot(
        plot_df_separate_flexible,
        x="feature_groups_short_name",
        y=metric_col,
        hue="model_name_hue",
        order=order,
        dodge=True,
        hue_order=hue_name.values(),
        # errorbar="pi",
    )  # , errorbar="pi")
    remove_legend = True
    # if remove_legend:
    #     g.get_legend().remove()
    # else:
    #     plt.legend(
    #         bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    #         loc="lower left",
    #         # ncols=2,
    #         mode="expand",
    #         borderaxespad=0.0,
    #     )
    if rotate_90:
        plt.xticks(rotation=90, ha="center")
    else:
        plt.xticks(rotation=45, ha="right")

    plt.legend(loc="upper left", title="Model")

    ylabel = metric_name
    if higher_better[metric_col] is not None:
        if higher_better[metric_col]:
            ylabel = ylabel + r" $\longrightarrow$"
        else:
            ylabel = r"$\longleftarrow$ " + ylabel

    plt.ylabel(ylabel)
    plt.xlabel("Patient Information")
    plt.grid(True)
    # plt.tight_layout()
    fig_name = f"separate_vs_flexible_model_{metric_col}"
    plt.savefig(FIGURE_PATH / (fig_name + ".pdf"), format="pdf", bbox_inches="tight")

    plt.show()

    if remove_legend:
        fig_legend = plt.figure()
        ax = fig_legend.gca()
        ax.legend(*g.get_legend_handles_labels(), loc="center", frameon=False)
        ax.axis("off")
        fig_legend.savefig(
            FIGURE_PATH / (fig_name + "_legend.pdf"), format="pdf", bbox_inches="tight"
        )
        fig_legend.show()
    break


# for fg in plot_df_separate_flexible["feature_groups_short_name"].unique():
#     dfs = plot_df_separate_flexible[
#         plot_df_separate_flexible["feature_groups_short_name"] == fg
#     ]
#     print(fg)
#     print(dfs.groupby("model_name")["roc_auc"].mean().sort_values(ascending=False))
#     print()


# mm = plot_df_separate_flexible.groupby(["model_name", "feature_groups_short_name"])[
#     "roc_auc"
# ].mean()

# mm = mm.reset_index()


# %%
# ------------------------ NEW COMBINATIONS ------------------------ #


plot_df_new_combs = plot_df[plot_df["group"].isin(["flexible_sampling_new_models"])]

# plot_df_new_combs = plot_df_new_combs[
#     ~plot_df_new_combs["feature_groups_short_name"].str.contains("\+")
# ]


plot_df_new_combs["mean_roc_auc"] = plot_df_new_combs.groupby(["directory"])[
    "roc_auc"
].transform("mean")


order = plot_df_new_combs.sort_values(by="mean_roc_auc", ascending=True)[
    "feature_groups_short_name"
].unique()


remove_legend = False
for metric_col, metric_name in relevant_metrics.items():
    plt.figure(figsize=(10, 6))

    g = sns.pointplot(
        plot_df_new_combs,
        x="feature_groups_short_name",
        y=metric_col,
        hue="model_name",
        order=order,
        dodge=True,
    )  # ,errorbar="pi",

    if remove_legend:
        g.get_legend().remove()
    else:
        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            # ncols=2,
            mode="expand",
            borderaxespad=0.0,
        )
    if rotate_90:
        plt.xticks(rotation=90, ha="center")
    else:
        plt.xticks(rotation=45, ha="right")
    ylabel = metric_name
    if higher_better[metric_col] is not None:
        if higher_better[metric_col]:
            ylabel = ylabel + r" $\longrightarrow$"
        else:
            ylabel = r"$\longleftarrow$ " + ylabel

    plt.ylabel(ylabel)

    plt.xlabel("Patient Information")
    # show grid

    # plt.tight_layout()
    fig_name = f"flexible_model_new_combinations_{metric_col}"
    plt.savefig(FIGURE_PATH / (fig_name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.show()

    if remove_legend:
        fig_legend = plt.figure()
        ax = fig_legend.gca()
        ax.legend(*g.get_legend_handles_labels(), loc="center", frameon=False)
        ax.axis("off")
        fig_legend.savefig(
            FIGURE_PATH / (fig_name + "_legend.pdf"), format="pdf", bbox_inches="tight"
        )
        fig_legend.show()
    break


# %%

evals_flexible = evals[
    evals["directory"].isin(plot_df_separate_flexible["directory"].unique())
]

evals_flexible["model_name"] = evals_flexible["directory"].map(
    plot_df_separate_flexible.set_index("directory")["model_name"].to_dict()
)

g = sns.FacetGrid(
    evals_flexible,
    col="model_name",
    col_wrap=3,
    height=3,
    aspect=1.5,
    sharex=False,
    sharey=False,
    hue="y_true",
    palette="two_color_palette",
    col_order=plot_df_separate_flexible["model_name"].unique(),
)
g.map(
    sns.histplot,
    "y_pred_score",
    bins=30,
    kde=True,
    fill=True,
    stat="density",
    common_norm=False,
    hue_order=[True, False],
    alpha=0.6,
)


# %%

# ------------------------ FULL PRECISION ------------------------ #

plot_df_full_precision = plot_df[plot_df["group"] == "precision_comp"]

plot_df_full_precision = pd.concat(
    [plot_df_full_precision, plot_df_base_risk_factors], axis=0
)

plot_df_full_precision["mean_roc_auc"] = plot_df_full_precision.groupby(["directory"])[
    "roc_auc"
].transform("median")

plot_df_full_precision.sort_values(by="mean_roc_auc", ascending=True, inplace=True)


plt.figure(figsize=(6, 4.5))
sns.boxplot(
    plot_df_full_precision,
    x="directory",
    y="roc_auc",
    hue="directory",
    fliersize=0,
)
plt.xticks(rotation=90, ha="center")
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc="lower left",
    # ncols=2,
    mode="expand",
    borderaxespad=0.0,
)

# %%

# compare evals

evals_full_precision = evals[
    evals["directory"].isin(plot_df_full_precision["directory"].unique())
]


g = sns.FacetGrid(
    evals_full_precision,
    col="directory",
    col_wrap=2,
    height=4,
    aspect=4,
    sharex=False,
    sharey=False,
    hue="y_true",
    palette="two_color_palette",
)
g.map(
    sns.histplot,
    "y_pred_score",
    bins=50,
    kde=True,
    fill=True,
    stat="density",
    common_norm=False,
    hue_order=[True, False],
    alpha=0.6,
)

plot_df_full_precision.groupby("directory")["model_name"].first()


# %%
# tune decision threshold

model_name = "Framingham Risk Score"

for model_name in plot_df["model_name"].unique():

    dir = plot_df[plot_df["model_name"] == model_name].directory.unique()[0]

    evals_x = evals[evals["directory"] == dir]

    evals_x.directory.unique()

    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(
        evals_x["y_true"], evals_x["y_pred_score"]
    )

    plt.figure(figsize=(5, 4))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.title(model_name)
    plt.show()

    # plot positive vs negative accuracy at different thresholds

    thresholds = np.linspace(0, 1, 100)
    positive_accuracy = []
    negative_accuracy = []

    for threshold in thresholds:
        y_pred = evals_x["y_pred_score"] > threshold
        y_true = evals_x["y_true"]
        positive_accuracy.append(np.mean(y_pred[y_true.astype(bool)]))
        negative_accuracy.append(np.mean(~y_pred[~y_true.astype(bool)]))

    plt.figure(figsize=(5, 4))
    plt.plot(thresholds, positive_accuracy, label="Positive Accuracy")
    plt.plot(thresholds, negative_accuracy, label="Negative Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title(model_name)
    plt.legend()
    plt.show()

    # %%


#
n = 14
m = 0.5
s = np.linspace(0, m, 100)

n_s = (n * s).astype(int)
n_s = np.ceil(n * s)

plt.figure(figsize=(5, 4))
plt.plot(s, n_s)
plt.hlines(1, 0, m, color="gray", linestyle="--")
plt.hlines(5, 0, m, color="gray", linestyle="--")

np.where(n_s == 2, s, 0)


# %%


plot_df["model_name"] = None
plot_df.loc[plot_df["directory"].str.contains("cox"), "model_name"] = "Cox PH"
plot_df.loc[plot_df["directory"].str.contains("lgbm"), "model_name"] = "LGBM"
plot_df.loc[plot_df["directory"].str.contains("mistral"), "model_name"] = "LLM"


plt.figure(figsize=(10, 6))
sns.boxplot(
    plot_df,
    x="data.num_training_samples",
    y="roc_auc",
    hue="model_name",
    fliersize=0,
)


plt.figure(figsize=(18, 6))
sns.boxplot(
    plot_df,
    x="data.num_training_samples",
    y="roc_auc",
    hue="feature_groups_short_name",
    fliersize=0,
)

# %%

# SURVIVAL CURVES / EVENT CURVES

full_model = "Base + PRS"  # "Base + ICD + PM + LE + FH + SD + PRS + MH + BS + UA"
selected_feature_groups = ["Base", full_model]

plot_df_surv = plot_df[
    (
        (plot_df["group"] == "mistral_all_fgs")
        & (
            plot_df["feature_groups_short_name"].isin(
                plot_df[plot_df["group"] == "mistral_all_fgs"][
                    "feature_groups_short_name"
                ].unique()
            )
        )
    )
    | (plot_df["model_name"].isin(plot_df_base_risk_factors["model_name"].unique()))
]


directories_surv = plot_df_surv["directory"].unique()

evals_survival_curves = evals[(evals["directory"].isin(directories_surv))]

evals_survival_curves.reset_index(drop=False, inplace=True)

model_name_mapping = plot_df_base_risk_factors.set_index("directory")[
    "model_name_short"
].to_dict()

plot_df_surv.loc[plot_df_surv["group"] == "mistral_all_fgs", "model_name_short"] = (
    plot_df_surv["model_name_short"]
    + " – "
    + plot_df_surv["feature_groups_short_name"].replace(
        full_model, "All Patient Information"
    )
)

model_name_mapping.update(
    plot_df_surv.set_index("directory")["model_name_short"].to_dict()
)

evals_survival_curves["model_name_short"] = evals_survival_curves["directory"].map(
    model_name_mapping
)

# x axis: days since baseline
# y axis: survival probability / share

evals_survival_curves["days_since_baseline"] = evals_survival_curves[
    "MACE_ADO_EXTENDED_days_from_baseline"
].fillna(365 * 10 + 100)

quantiles = [0, 0.90, 0.99, 1.0]
labels = [
    f"Lowest {int(quantiles[1] * 100)}%",
    f"{int(quantiles[1] * 100)}% - {int(quantiles[2] * 100)}%",
    f"Highest {int(100-quantiles[2] * 100)}%",
]

risk_color_mapping = {
    labels[0]: "#457b9d",
    labels[1]: "#a8dadc",
    labels[2]: "#e63946",
}

evals_survival_curves["Risk Quantile"] = evals_survival_curves.groupby("directory")[
    "y_pred_score"
].transform(lambda x: pd.qcut(x, q=quantiles, labels=labels))

col_order = evals_survival_curves["model_name_short"].unique()

only_few_models = False
if only_few_models:
    models = [
        "Fine-Tuned LLM Mistral-7B – Base",
        "Fine-Tuned LLM Mistral-7B – All Patient Information",
    ]
    evals_survival_curves = evals_survival_curves[
        evals_survival_curves["model_name_short"].isin(models)
    ]
    col_order = models  # + ["Fine-Tuned LLM (Best feature set)"]


evals_survival_curves["y_true"] = evals_survival_curves["y_true"].astype(bool)

# distribution of predictions
g = sns.FacetGrid(
    evals_survival_curves,
    col="model_name_short",
    col_wrap=1 if only_few_models else 5,
    height=2,
    aspect=1.5,
    sharex=True,
    sharey=False,
    hue="y_true",
    palette="two_color_palette",
    # col_order=col_order,
)

g.map(
    sns.kdeplot,
    "y_pred_score",
    # bins=30,
    # kde=True,
    fill=True,
    # stat="density",
    common_norm=False,
    hue_order=[True, False],
    alpha=0.5,
    bw_adjust=1.2,
)
g.set_titles("{col_name}", size=9)

g.add_legend(
    title="Observed Outcome",
    loc="upper center",
    bbox_to_anchor=(0.0, 1.05, 1.0, 1.05),
    ncol=2,
    labelspacing=0.5,
    columnspacing=1.0,
)

g.set_axis_labels("Prediction", "Density")

plt.savefig(
    FIGURE_PATH / f"distribution_of_predictions{'_few' if only_few_models else ''}.pdf",
    format="pdf",
    bbox_inches="tight",
)

g = sns.FacetGrid(
    evals_survival_curves,
    col="model_name_short",
    col_wrap=1 if only_few_models else 5,
    height=2,
    aspect=1.8,
    # col_order=col_order,
    palette=risk_color_mapping,
    hue="Risk Quantile",
    sharex=False,
    sharey=False,
)

g.map(
    sns.ecdfplot,
    "days_since_baseline",
    stat="percent",
    # complementary=True,
    palette=risk_color_mapping,
    # linewidth=2,
)

g.set(ylim=(0, 20), xlim=(0, 365 * 10))
g.set_titles("{col_name}", size=9)

g.set_axis_labels("Days since Baseline Assessment", "Percent")

g.add_legend(
    title="Risk Quantile",
    ncol=3,
    labelspacing=0.5,
    columnspacing=1.0,
    loc="upper center",
    bbox_to_anchor=(0.0, 1.05, 1.0, 1.05),
)

# remove grid


# g.fig.suptitle("Distribution of Risk Predictions", y=1.01)

# ax.grid(False)

plt.savefig(
    FIGURE_PATH / f"survival_curves{'_few' if only_few_models else ''}.pdf",
    format="pdf",
    bbox_inches="tight",
)

# calibration plot
bins = np.linspace(0, 1, 20)
evals_survival_curves["bin"] = pd.cut(evals_survival_curves["y_pred_score"], bins=bins)
calibration = evals_survival_curves.groupby(["bin", "model_name_short"])[
    ["y_pred_score", "y_true"]
].mean()
calibration[evals_survival_curves.groupby(["bin", "model_name_short"]).size() < 20] = (
    np.nan
)

# Define figure size
w = 0.3 * textwidth
h = w

# Create the main calibration plot
plt.figure(figsize=(w, h))
g = sns.lineplot(
    data=calibration,
    x="y_pred_score",
    y="y_true",
    hue="model_name_short",
    style="model_name_short",
    markers=True,
    # legend=False,  # Remove legend from the main plot
)

# plt.title("Calibration Plot")
plt.xlabel("Predicted Risk")
plt.ylabel("Observed Risk")

# Reference diagonal line
plt.plot([0, 1], [0, 1], color="grey", linestyle="--")

gridlines = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
plt.xticks(gridlines)
plt.yticks(gridlines)

remove_legend = True
if remove_legend:
    g.get_legend().remove()
else:
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        # ncols=2,
        mode="expand",
        borderaxespad=0.0,
    )

# Save the main plot
fig_name = f"calibration_plot{'_few' if only_few_models else ''}"
if save:
    plt.savefig(
        FIGURE_PATH / f"{fig_name}.pdf",
        bbox_inches="tight",
        format="pdf",
    )
plt.show()

# Create a separate figure for the legend
fig_legend, ax_legend = plt.subplots(
    figsize=(w * 0.6, h * 0.3)
)  # Adjust size for the legend only
handles, labels = g.get_legend_handles_labels()
ax_legend.legend(handles, labels, loc="center", frameon=True)
ax_legend.axis("off")  # Remove axis

# Save the legend separately
legend_name = "calibration_legend"
if save:
    fig_legend.savefig(
        FIGURE_PATH / f"{legend_name}.pdf",
        bbox_inches="tight",
        format="pdf",
    )

plt.show()


# %%
# SUBGROUPS

selected_feature_groups = [
    "Base",
    "Base + ICD + PM + LE + FH + SD + PRS + MH + BS + UA",
]

models = [
    "Framingham Risk Score",
    "PREVENT Risk Score",
    # "Cox PH Model",
    # "LGBM",
    "Fine-Tuned LLM Mistral-7B",
]


# retrieve subgroups
dir_to_subgroup = {
    s: s.split("subgroups/")[-1] for s in plot_df_sg["directory"].unique()
}
plot_df_sg["subgroup_and_value"] = plot_df_sg["directory"].map(dir_to_subgroup)
plot_df_sg["subgroup"] = plot_df_sg["subgroup_and_value"].apply(
    lambda x: x.split("/")[0]
)
plot_df_sg["subgroup_value"] = plot_df_sg["subgroup_and_value"].apply(
    lambda x: x.split("/")[1]
)

# %%

subgroups = plot_df_sg["subgroup"].unique()
subgroup_name_mapping = {
    "Sex": {
        "name": "Gender",
        "values": {"Male": "Male", "Female": "Female"},
    },
    "Age_Group": {
        "name": "Age Group",
        "values": {"(38,_50]": "38-50", "(50,_60]": "50-60", "(60,_70]": "60-70"},
    },
    "BMI_Group": {
        "name": "BMI Group",
        "values": {"Normal": "Normal", "Overweight": "Overweight", "Obese": "Obese"},
    },
    "Diabetes": {
        "name": "Diabetes",
        "values": {"No": "No", "Yes": "Yes"},
    },
    "Smoking_Status": {
        "name": "Smoking Status",
        "values": {"Never": "Never", "Previous": "Previous", "Current": "Current"},
    },
    "Average_Household_Income": {
        "name": "Average Household Income",
        "values": {
            "Prefer_not_to_answer": "Prefer not to answer",
            # "Do_not_know": "Do not know",
            "18,000_to_30,999": "18.000-30.999",
            "31,000_to_51,999": "31.000-51.999",
            "52,000_to_100,000": "52-000-100.000",
            "Greater_than_100,000": "100.000+",
        },
    },
    "Home_Area_Category": {
        "name": "Home Area Population Density",
        "values": {
            "Urban": "Urban",
            "Suburban_&_Small_Town": "Suburban & Small Town",
            "Rural": "Rural",
        },
    },
    "Private_Healthcare_Grouped": {
        "name": "Private Healthcare",
        "values": {"Yes": "Yes", "No": "No"},
    },
    "Qualifications_Grouped": {
        "name": "Education",
        "values": {
            "Higher_Education": "Higher Education",
            "Secondary_Education": "Secondary Education",
            "No_Formal_Education": "No Formal Education",
        },
    },
    "Shift_Work_Category": {
        "name": "Job Involves Shift Work",
        "values": {"Yes": "Yes", "No": "No"},
    },
}
# TODO: highest qualification

cpal_subgroups = sns.color_palette("custom_palette", n_colors=len(models))
sns.set_palette(palette=cpal_subgroups)


plot_df_sg["model_name_short_features"] = plot_df_sg["model_name"].copy()

plot_df_sg.loc[
    plot_df_sg["group"].isin(["mistral_all_fgs"]),
    "model_name_short_features",
] = (
    "AdaCVD"  # plot_df_sg["model_name_short_features"]
    + " – "
    + plot_df_sg["feature_groups_short_name"]
)


# n_plus = plot_df_sg["feature_groups_short_name"].str.count("\+")
# plot_df_sg.loc[n_plus == 9, "feature_groups_display_name"] = (
#     "All Patient Information"
# )

#    "#E73EB3",  # pink
#     "#874CD6",  # lila
#     "#2B0091",  # dark blue
#     "#006FF2",  # blue
#     "#00D2FF",  # light blue
#     "#4AFCE2",  # turquoise
#     "#6CFFCA",  # light green
#     "#A3FF32",  # green
#     "#F9F871",  # yellow

hue_order = {
    "Framingham Risk Score": category_color_dict["Risk Scores"],
    "PREVENT Risk Score": category_color_dict["ML Models"],
    # "LGBM – Base": category_color_dict["ML Models"],
    "AdaCVD – Base": category_color_dict["AdaCVD"],
    # "Fine-Tuned LLM Mistral-7B – Base + UA": "navy",
    # "Fine-Tuned LLM Mistral-7B – Base + SD": "navy",
    # "Fine-Tuned LLM Mistral-7B – Base + PM": "#4895ef",
    # "Fine-Tuned LLM Mistral-7B – Base + FH": "navy",
    # "Fine-Tuned LLM Mistral-7B – Base + BS": "navy",
    "AdaCVD – Base + LE": color_palette.BLUE[2],
    "AdaCVD – Base + PRS": color_palette.BLUE[3],
    # "AdaCVD – Base + ICD": "#006FF2",
    "AdaCVD – Base + MH": color_palette.TEAL[1],
    # "LGBM – Base + ICD + PM + LE + FH + SD + PRS + MH + BS + UA": "navy",
    "AdaCVD – All Patient Information": color_palette.TEAL[0],
}

plot_df_sg["model_name_short_features"] = plot_df_sg["model_name_short_features"].apply(
    lambda x: x.replace(
        "Base + ICD + PM + LE + FH + SD + PRS + MH + BS + UA",
        "All Patient Information",
    )
)

plot_df_sg_subset = plot_df_sg[
    plot_df_sg["model_name_short_features"].isin(hue_order.keys())
]


# subgroup_color_dict = {
#     "PREVENT Risk Score": category_color_dict["Risk Scores"],
#     "Cox PH Model": category_color_dict["ML Models"],
#     "Gradient Boosted Trees": category_color_dict["ML Models"],
#     "Fine-Tuned LLM Mistral-7B": category_color_dict["AdaCVD"],
# }


relative_increases = {}

kwargs = dict(palette=hue_order)  # TODO


for subgroup, subgroup_info in subgroup_name_mapping.items():
    for metric_col, metric_name in relevant_metrics.items():
        plot_df_sg_subgroup = plot_df_sg[plot_df_sg["subgroup"] == subgroup]

        plot_df_subset = plot_df[
            plot_df["group"].isin(plot_df_sg_subgroup.group.unique())
        ]

        if len(plot_df_sg_subgroup.group.unique()) == 1:
            continue

        # plot_df_subset["subgroup_value"] = "All"
        # plot_df_sg_subgroup = pd.concat([plot_df_sg_subgroup, plot_df_subset], axis=0)

        # plot_df_sg_subgroup = plot_df_sg_subgroup[
        #     plot_df_sg_subgroup["model_name"].str.contains("LGBM")
        # ]

        x = "subgroup_value"
        x_order = [*subgroup_info["values"].keys()]
        x_labels = [*subgroup_info["values"].values()]

        # hue_order = (
        #     plot_df_sg_subgroup.groupby("model_name_short_features")[metric_col]
        #     .median()
        #     .sort_values()
        #     .index,
        # )
        for subgroup_value in x_order:
            tmp_df = plot_df_sg_subgroup[
                plot_df_sg_subgroup["subgroup_value"] == subgroup_value
            ]

            base_median = tmp_df[
                tmp_df["model_name_short_features"] == "AdaCVD – Base"
            ][metric_col].median()
            print(base_median)
            print(tmp_df.groupby("model_name_short_features")[metric_col].median())
            print(subgroup)
            print(subgroup_value)
            rel = (
                (
                    tmp_df.groupby("model_name_short_features")[metric_col].median()[
                        "AdaCVD – All Patient Information"
                    ]
                    - base_median
                )
                / base_median
                * 100
            )
            print(rel)

            relative_increases[(subgroup, subgroup_value)] = rel

        w = 0.45 * textwidth
        h = 0.5 * w
        plt.figure(figsize=(w, h))

        boxplot_args = dict(
            data=plot_df_sg_subgroup,
            x=x,
            y=metric_col,
            hue="model_name_short_features",
            hue_order=hue_order.keys(),
            order=x_order,
            fliersize=0.1,
            saturation=1.0,
            **kwargs,
            dodge=True,
            gap=0.15,
        )

        g = sns.boxplot(
            **boxplot_args,
            linewidth=0,
            showfliers=False,
            boxprops=dict(alpha=0.5),
        )

        sns.boxplot(**boxplot_args, fill=False, legend=False)
        remove_legend = True
        if remove_legend:
            g.get_legend().remove()
        else:
            plt.legend(
                bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
                loc="lower left",
                # ncols=2,
                mode="expand",
                borderaxespad=0.0,
            )

        rotation = 0
        if max([len(str(x)) for x in x_labels]) > 10:
            rotation = 20
        plt.xticks(rotation=rotation, ha="center", ticks=x_order, labels=x_labels)
        # plt.xticks(ticks=x_order, labels=x_labels)

        if metric_col == "roc_auc":
            plt.ylim(0.55, 0.85)

        ylabel = metric_name
        if higher_better[metric_col] is not None:
            if higher_better[metric_col]:
                ylabel = ylabel + r" $\longrightarrow$"
            else:
                ylabel = r"$\longleftarrow$ " + ylabel

        plt.ylabel(ylabel)

        plt.xlabel(subgroup_info["name"], labelpad=ylabelpad),  # TODO: Name
        # plt.tight_layout()
        fig_name = f"subgroups_{subgroup}_{metric_col}"
        plt.savefig(
            FIGURE_PATH / (fig_name + ".pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

        if remove_legend:

            g = sns.boxplot(
                **boxplot_args,
                linewidth=0,
                showfliers=False,
            )
            plt.close()

            fig_legend = plt.figure(figsize=(w, h))
            ax = fig_legend.gca()
            ax.legend(
                *g.get_legend_handles_labels(),
                loc="center",
                frameon=False,
                ncol=4,  # int(len(g.get_legend_handles_labels()[1]) / 3),
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
        break

    # break

# %%
# to dataframe (relative_increases)

relative_increases_df = pd.DataFrame(
    relative_increases.items(), columns=["subgroup", "relative_increase"]
)

relative_increases_df.sort_values(by="relative_increase", ascending=False)


# %%
# efficiency

plot_df_efficiency = plot_df[
    plot_df["group"].isin(["efficiency", "efficiency_more_features"])
]

plot_df_efficiency["model_name"] = None
plot_df_efficiency.loc[
    plot_df_efficiency["directory"].str.contains("cox"), "model_name"
] = "Cox PH"
plot_df_efficiency.loc[
    plot_df_efficiency["directory"].str.contains("lgbm"), "model_name"
] = "LGBM"

x = "data.num_training_samples"
y = "roc_auc"
hue = "feature_groups_short_name"
style = "model_name"

g = sns.FacetGrid(
    data=plot_df_efficiency, col=style, col_wrap=2, height=5, aspect=2, hue=hue
)
g.map(sns.pointplot, x, y, dodge=True, markersize=8, marker="o")

# legend
g.add_legend(
    title="Feature Groups",
    ncol=1,
    labelspacing=0.5,
    columnspacing=1.0,
    loc="upper center",
    bbox_to_anchor=(0.0, 1.05, 1.0, 1.05),
)

plt.savefig(
    FIGURE_PATH / f"efficiency_lgbm_cox.pdf",
    format="pdf",
    bbox_inches="tight",
)

plt.figure(figsize=(6, 4.5))
sns.pointplot(plot_df_efficiency, x=x, y=y, hue=hue, style=style)


plot_df_efficiency[[x, y, hue, style]].isna().sum()

# %%
# INFERENCE FORMAT TRANSFORMED

g1 = "format_inference_orig"  # "base_llm_mistral"
g2 = "format_inference"


plot_df_format = plot_df[plot_df["group"].isin([g1, g2])]

plot_df_format.groupby("group")["roc_auc"].median()

evals_format = evals[evals["group"].isin(plot_df_format["group"].unique())]

evals_format_wide = evals_format.reset_index().pivot(
    index="eid", columns="group", values=["y_pred_score", "y_true"]
)

evals_format_wide = evals_format_wide.dropna()

plt.figure(figsize=(6, 4.5))
sns.histplot(evals_format, x="y_pred_score", hue="group", bins=10)

evals_format_wide["diff"] = (
    evals_format_wide["y_pred_score", g1] - evals_format_wide["y_pred_score", g2]
)

plt.figure(figsize=(6, 4.5))
sns.histplot(evals_format_wide, x=("diff", ""), bins=30)
plt.xlim(-0.2, 0.2)

# boxplot of roc_auc
plt.figure(figsize=(6, 4.5))
sns.boxplot(
    plot_df_format,
    x="directory",
    y="roc_auc",
    fliersize=0,
)
plt.xticks(rotation=90, ha="center")


plot_df_format.groupby("directory")["roc_auc"].median().sort_values()

# %%

# FEATURE GROUPS

fg_info = pd.read_csv("feature_group_info.csv")

groups = [
    "urine_assays",
    "sociodemographics",
    "physical_measures",
    "family_history",
    "blood_samples",
    "lifestyle_and_environment",
    "polygenic_risk_scores_all",
    "icd_codes",
    "medical_history_all",
    "all",
]

group_to_name = {"all_base_risk_factors": "Base Risk Factors"}

for g in groups:
    if g in name_mapping.feature_group_names:
        group_to_name[g] = name_mapping.feature_group_names[g]["long_name"]

group_to_name["all"] = "All Patient Information"

fg_info["feature_group_long_name"] = fg_info["feature_group"].map(group_to_name)

fg_info["lower"] = fg_info["num_tokens_median"] - fg_info["num_tokens_lower_05"]
fg_info["upper"] = fg_info["num_tokens_upper_05"] - fg_info["num_tokens_median"]

fg_info_ordered = fg_info.set_index("feature_group_long_name").loc[
    list(group_to_name.values())
]

x_err = fg_info_ordered[["lower", "upper"]].T

# w = 0.17 * textwidth
# h = 1.4 * w
# plt.figure(figsize=(w, h))

# # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(w, h))
# # fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
# colors = [palette[x] for x in group_to_name.values()]

# sns.barplot(
#     data=fg_info,
#     x="num_tokens_median",
#     y="feature_group_long_name",
#     order=group_to_name.values(),
#     # palette=colors,
#     color=color_dict["Base + Single Group"],
#     xerr=x_err,
#     ecolor="black",
#     # palette=colors,
# )

# plt.xlabel("Number of Tokens")
# plt.ylabel("Feature Group")

# plt.savefig(
#     FIGURE_PATH / "feature_groups.pdf",
#     format="pdf",
#     bbox_inches="tight",
# )

# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define figure dimensions
w = 0.195 * textwidth
h = 1.4 * w
fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, figsize=(w, h), gridspec_kw={"width_ratios": [1, 1]}
)

# Adjust space between the two axes
fig.subplots_adjust(wspace=0.1)

# Define x-axis limits for each subplot (adjust as needed)
xlim1 = (0, 1100)  # Lower range
xlim2 = (2400, 3500)  # Upper range


# Plot the same data on both axes
sns.barplot(
    data=fg_info,
    x="num_tokens_median",
    y="feature_group_long_name",
    order=group_to_name.values(),
    color=color_dict["Base + Single Group"],
    xerr=x_err,
    ecolor="black",
    ax=ax1,
)
sns.barplot(
    data=fg_info,
    x="num_tokens_median",
    y="feature_group_long_name",
    order=group_to_name.values(),
    color=color_dict["Base + Single Group"],
    xerr=x_err,
    ecolor="black",
    ax=ax2,
)

# Set x-axis limits for each subplot
ax1.set_xlim(xlim1)
ax2.set_xlim(xlim2)

# Add broken-axis diagonal markers
# gridcolor
gridcolor = plt.rcParams["grid.color"]

d = 0.015  # Marker size
kwargs = dict(transform=ax1.transAxes, color=gridcolor, clip_on=False)
ax1.plot([1 - d, 1 + d], [-d, +d], **kwargs)  # Bottom diagonal line
ax1.plot([1 - d, 1 + d], [1 - d, 1 + d], **kwargs)  # Top diagonal line

kwargs.update(transform=ax2.transAxes)
ax2.plot([-d, +d], [-d, +d], **kwargs)  # Bottom diagonal line
ax2.plot([-d, +d], [1 - d, 1 + d], **kwargs)  # Top diagonal line

ax1.spines.right.set_visible(False)
# ax1.spines.top.set_visible(False)
# ax2.spines.top.set_visible(False)
ax2.spines.left.set_visible(False)
ax1.tick_params(labeltop=False)  # don't put tick labels at the top

# remove top axis
ax1.xaxis.tick_bottom()

ax2.xaxis.tick_bottom()


# Set labels
ax1.set_xlabel("Number of Tokens")
ax1.set_ylabel("Feature Group")

# remove ax2 xlabel
ax2.set_xlabel("")

# rotate labels 90 degrees
plt.setp(ax1.get_xticklabels(), rotation=90, ha="center")
plt.setp(ax2.get_xticklabels(), rotation=90, ha="center")


# Save figure
plt.savefig(
    FIGURE_PATH / "feature_groups.pdf",
    format="pdf",
    bbox_inches="tight",
)

# %%
# compare lenghts of prompts between different dataset

lengths_text = {}

# UKB-text
ukb_text_base = pd.read_csv(
    "/Users/frederike/Documents/PhD/BioBank/code/2025_04_02_text_datasets/2025_02_26_text_dataset_base_mistral_20k/wandb_export_2025-04-02T11_53_13.617+02_00.csv"
)

ukb_text_full = pd.read_csv(
    "/Users/frederike/Documents/PhD/BioBank/code/2025_04_02_text_datasets/2025_02_26_text_dataset_full_mistral_20k/wandb_export_2025-04-02T11_54_41.864+02_00.csv"
)

lengths_text["ukb_text_base"] = {
    "mean": ukb_text_base[
        "2025-02-26_12-57-44-417873 - num_generated_tokens_without_eos_mean"
    ].mean(),
    "lower": ukb_text_base[
        "2025-02-26_12-57-44-417873 - num_generated_tokens_without_eos_mean"
    ].min(),
    "upper": ukb_text_base[
        "2025-02-26_12-57-44-417873 - num_generated_tokens_without_eos_mean"
    ].max(),
}


lengths_text["ukb_text_full"] = {
    "mean": ukb_text_full[
        "2025-02-26_15-44-54-649153 - num_generated_tokens_without_eos_mean"
    ].mean(),
    "lower": ukb_text_full[
        "2025-02-26_15-44-54-649153 - num_generated_tokens_without_eos_mean"
    ].min(),
    "upper": ukb_text_full[
        "2025-02-26_15-44-54-649153 - num_generated_tokens_without_eos_mean"
    ].max(),
}

# UKB tab

# for feature_group in group_to_name.values():
#     lengths[feature_group] = {
#         "median": fg_info[fg_info["feature_group"] == feature_group][
#             "num_tokens_median"
#         ].values[0],
#         "min": fg_info[fg_info["feature_group"] == feature_group][
#             "num_tokens_lower_05"
#         ].values[0],
#         "max": fg_info[fg_info["feature_group"] == feature_group][
#             "num_tokens_upper_05"
#         ].values[0],
#     }

# Next Steps: Plot this as barplot, add framingham
# Write a separate script that loads them, tokenizes, and calculates the lengths (min, median, max, quantiles)
# And for now, just do it manually (CSV)

# add fg_info.loc[fg_info["feature_group"] == "all_base_risk_factors", ["num_tokens_median", "num_tokens_lower_05", "num_tokens_upper_05"]] to all other rows


if (
    fg_info[fg_info["feature_group"] == "polygenic_risk_scores_all"]["num_tokens_mean"]
    == 731.7179300291546
):

    fg_info.loc[
        fg_info["feature_group"] != "all_base_risk_factors", "num_tokens_mean"
    ] += fg_info[fg_info["feature_group"] == "all_base_risk_factors"][
        "num_tokens_mean"
    ].values[
        0
    ]

    fg_info.loc[
        fg_info["feature_group"] != "all_base_risk_factors", "num_tokens_lower_05"
    ] += fg_info[fg_info["feature_group"] == "all_base_risk_factors"][
        "num_tokens_lower_05"
    ].values[
        0
    ]

    fg_info.loc[
        fg_info["feature_group"] != "all_base_risk_factors", "num_tokens_upper_05"
    ] += fg_info[fg_info["feature_group"] == "all_base_risk_factors"][
        "num_tokens_upper_05"
    ].values[
        0
    ]


lengths = (
    fg_info[
        [
            "feature_group_long_name",
            "num_tokens_mean",
            "num_tokens_lower_05",
            "num_tokens_upper_05",
        ]
    ]
    .set_index("feature_group_long_name")
    .copy()
)

lengths["dataset"] = "UKB-tab"

lengths = lengths.reset_index()

lengths = lengths.rename(
    columns={
        "num_tokens_mean": "mean",
        "num_tokens_lower_05": "lower",
        "num_tokens_upper_05": "upper",
    }
)

# add lenghts text to lengths
lengths_text_df = pd.DataFrame(lengths_text).T
lengths_text_df["dataset"] = "UKB-text"
lengths_text_df["feature_group_long_name"] = lengths_text_df.index.map(
    {
        "ukb_text_base": "Summary (Base Risk Factors)",
        "ukb_text_full": "Summary (All Patient Information)",
    }
)
lengths_text_df = lengths_text_df.reset_index(drop=True)
lengths = pd.concat([lengths, lengths_text_df], axis=0)

# Framingham
# TODO: update with correct numbers
lengths_framingham = {
    "mean": 135,
    "lower": 97,
    "upper": 145,
    "dataset": "Framingham",
    "feature_group_long_name": "Framingham Risk Factors",
}
lengths_framingham = pd.DataFrame(lengths_framingham, index=[0])

lengths = pd.concat([lengths, lengths_framingham], axis=0)
# %%
# barplot
w = 0.8 * textwidth
h = 0.35 * w

plt.figure(figsize=(w, h))

order = [
    *group_to_name.values(),
    "Summary (Base Risk Factors)",
    "Summary (All Patient Information)",
    "Framingham Risk Factors",
]

# # remove "all" from order
# order.remove("all")

# # remove all
# lengths = lengths[lengths["feature_group"] != "all"]

g = sns.barplot(
    data=lengths,
    x="feature_group_long_name",
    y="mean",
    hue="dataset",
    order=order,
    capsize=0.2,
    palette={"UKB-tab": "#457b9d", "UKB-text": "#a8dadc", "Framingham": "#e63946"},
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Feature Group")
plt.ylabel("Length of Patient Description (Tokens)")

# logscale (y)
# plt.yscale("log")

if remove_legend:
    g.get_legend().remove()

fig_name = "dataset_info"

plt.savefig(
    FIGURE_PATH / (fig_name + ".pdf"),
    format="pdf",
    bbox_inches="tight",
)


# legend in separate plot
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

# TODO: add split axis
# %%

# Define figure dimensions
w = 0.75 * textwidth
h = 0.3 * w
fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(w, h), gridspec_kw={"height_ratios": [0.3, 1]}
)

# Adjust space between the two axes
fig.subplots_adjust(hspace=0.1)

ylim1 = (0, 1100)  # Lower range
ylim2 = (2950, 3300)  # Upper range

# custom_palette = {
#     "AdaCVD (Feature Expert Models)": "#A0A0A0",  # "blue"
#     "AdaCVD-Flex (Single Model)": "#f66058",
#     "AdaCVD (All Patient Info; Single Model)": "#234875",  # "#415a77",
#     # # "LGBM: Separate Models": "#A0A0A0",
#     "LGBM (All Patient Info; Single Model)": "#667798",  # "#5d5d5d",
# }

palette = {"UKB-tab": "#234875", "UKB-text": "#5a78a9", "Framingham": "#a4abbd"}


# Plot the same data on both axes
sns.barplot(
    data=lengths,
    x="feature_group_long_name",
    y="mean",
    hue="dataset",
    order=order,
    capsize=0.2,
    palette=palette,
    ax=ax1,
)
sns.barplot(
    data=lengths,
    x="feature_group_long_name",
    y="mean",
    hue="dataset",
    order=order,
    capsize=0.2,
    palette=palette,
    ax=ax2,
    # clip_on=False
)

ax1.set_ylim(ylim2)
ax2.set_ylim(ylim1)

# Add broken-axis diagonal markers
# gridcolor
gridcolor = plt.rcParams["grid.color"]

d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color=gridcolor,
    # mec="k",
    mew=1,
    clip_on=False,
)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
# ax1.tick_params(labelbottom=False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax2.set_ylabel("Length of Patient Description")
ax1.set_ylabel("")

# remove legends
if remove_legend:
    ax1.get_legend().remove()
    ax2.get_legend().remove()

rotation = 60
plt.setp(ax1.get_xticklabels(), rotation=rotation, ha="right")
plt.setp(ax2.get_xticklabels(), rotation=rotation, ha="right")

fig_name = "dataset_info"

# remove gridlines
ax1.grid(False)
ax2.grid(False)

plt.xlabel("")

plt.savefig(
    FIGURE_PATH / (fig_name + ".pdf"),
    format="pdf",
    bbox_inches="tight",
)


# legend in separate plot
fig_legend = plt.figure(figsize=(w, h))
ax = fig_legend.gca()
ax.legend(
    *ax1.get_legend_handles_labels(),
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

# %%
