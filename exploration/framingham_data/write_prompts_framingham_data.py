import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from IPython.display import display

from exploration.frederike.visualization.visualization_utils import setup_plotting
from pandora.data import ukb_data_utils, ukb_features, ukb_field_ids
from pandora.risk_scores import (
    framingham_risk_score,
    pooled_cohort_risk_score,
    prevent_risk_score,
)
from pandora.utils.metrics import (
    compute_binary_classification_metrics,
    compute_bootstrapped_metrics,
    compute_roc_curve_values,
)

df = pd.read_csv(ukb_data_utils.ASSETS_PATH / "framingham/frmgham2.csv")
df = df.set_index("RANDID")


# UTILS
df["STRK_MI"] = df["MI_FCHD"] | df["STROKE"]
df["TIME_STRK_MI"] = df[["TIMEMIFC", "TIMESTRK"]].min(axis=1)
df["PREVCVD"] = (
    df["PREVCHD"] | df["PREVMI"] | df["PREVSTRK"]
)  # df["PREVAP"] # TODO: include AP and coronary heart disease or not?
df["PREVCVD_AP"] = df["PREVCHD"] | df["PREVMI"] | df["PREVSTRK"] | df["PREVAP"]

df["PREV_STRK_MI"] = df["PREVMI"] | df["PREVSTRK"]
df["CVD_STRK_MI"] = df["CVD"] | df["STROKE"] | df["MI_FCHD"]
df["TIME_CVD_STRK_MI"] = df[["TIMECVD", "TIMESTRK", "TIMEMIFC"]].min(axis=1)

baseline_assessment = df[df["PERIOD"] == 3].copy()


CHOLESTEROL_MMOLL_TO_MGDL_FACTOR = 38.67

prompt_parts = pd.DataFrame(index=baseline_assessment.index)
tab_df = pd.DataFrame(index=baseline_assessment.index)

mask = pd.Series(index=baseline_assessment.index)


prompt_parts["SEX"] = "Sex: " + baseline_assessment["SEX"].map({1: "Male", 2: "Female"})
tab_df["SEX"] = baseline_assessment["SEX"].map({1: "Male", 2: "Female"})
prompt_parts["AGE"] = "Age: " + baseline_assessment["AGE"].astype(str) + " years"
tab_df["AGE"] = baseline_assessment["AGE"]
prompt_parts["TOTCHOL"] = (
    "Cholesterol: " + baseline_assessment["TOTCHOL"].astype(str) + " mg/dL"
)
tab_df["TOTCHOL"] = baseline_assessment["TOTCHOL"]
prompt_parts["TOTCHOL_2"] = (
    "Cholesterol: "
    + round(
        baseline_assessment["TOTCHOL"] / CHOLESTEROL_MMOLL_TO_MGDL_FACTOR, 1
    ).astype(str)
    + " mmol/L"
)
tab_df["TOTCHOL_2"] = baseline_assessment["TOTCHOL"] / CHOLESTEROL_MMOLL_TO_MGDL_FACTOR

# HDLC
prompt_parts["HDLC"] = (
    "HDL Cholesterol: " + baseline_assessment["HDLC"].astype(str) + " mg/dL"
)
tab_df["HDLC"] = baseline_assessment["HDLC"]

prompt_parts["HDLC_2"] = (
    "HDL Cholesterol: "
    + round(baseline_assessment["HDLC"] / CHOLESTEROL_MMOLL_TO_MGDL_FACTOR, 1).astype(
        str
    )
    + " mmol/L"
)
tab_df["HDLC_2"] = baseline_assessment["HDLC"] / CHOLESTEROL_MMOLL_TO_MGDL_FACTOR

# HDLC: for Period != 3, simple mean imputation from paper
# prompt_parts["HDLC"] = (
#     "HDL cholesterol: "
#     + baseline_assessment["SEX"].map({1: 44.9, 2: 57.6}).astype(str)
#     + " mg/dL"
# )
# prompt_parts["HDLC_2"] = (
#     "HDL cholesterol: "
#     + round(
#         baseline_assessment["SEX"].map({1: 44.9, 2: 57.6})
#         / CHOLESTEROL_MMOLL_TO_MGDL_FACTOR,
#         1,
#     ).astype(str)
#     + " mmol/L"
# )


prompt_parts["SYSBP"] = (
    "Systolic blood pressure: " + baseline_assessment["SYSBP"].astype(str) + " mmHg"
)
tab_df["SYSBP"] = baseline_assessment["SYSBP"]
prompt_parts["DIABETES"] = "Diabetes diagnosed by doctor: " + baseline_assessment[
    "DIABETES"
].map({0: "No", 1: "Yes"})
tab_df["DIABETES"] = baseline_assessment["DIABETES"].map({0: "No", 1: "Yes"})
prompt_parts["CURSMOKE"] = "Smoking: " + baseline_assessment["CURSMOKE"].map(
    {0: "No", 1: "Yes"}
)
tab_df["CURSMOKE"] = baseline_assessment["CURSMOKE"].map({1: "Current", 0: "Never"})
prompt_parts["BPMEDS"] = "Blood pressure medication: " + baseline_assessment[
    "BPMEDS"
].map({0: "No", 1: "Yes"})
tab_df["BPMEDS"] = baseline_assessment["BPMEDS"].astype(bool)
prompt_parts["BMI"] = "BMI: " + baseline_assessment["BMI"].astype(str)
tab_df["BMI"] = baseline_assessment["BMI"]

for col in prompt_parts.columns:
    prompt_parts[col] = prompt_parts[col].apply(
        lambda x: "" if isinstance(x, str) and "nan" in x.lower() else x
    )
    prompt_parts[col] = prompt_parts[col].apply(lambda x: "" if pd.isna(x) else x)

mask = tab_df.isna().max(axis=1)
tab_df = tab_df.loc[~mask]
prompt_parts = prompt_parts.loc[~mask]


target_options = {
    "CVD": {
        "target_variable": "CVD",
        "time_target_variable": "TIMECVD",
        "prevalence_variable": "PREV_STRK_MI",  # TODO: why not PREVCVD or PREVCVD_AP?
    },
    "ANYCHD": {
        "target_variable": "ANYCHD",
        "time_target_variable": "TIMECHD",
        "prevalence_variable": "PREVCHD",
    },
    "CVD_STRK_MI": {
        "target_variable": "CVD_STRK_MI",
        "time_target_variable": "TIME_CVD_STRK_MI",
        "prevalence_variable": "PREV_STRK_MI",
    },
    "STRK_MI": {
        "target_variable": "STRK_MI",
        "time_target_variable": "TIME_STRK_MI",
        "prevalence_variable": "PREV_STRK_MI",
    },
}
for unit in ["mgdl", "mmoll"]:
    for target_name, target_def in target_options.items():
        target_variable = target_def["target_variable"]
        time_target_variable = target_def["time_target_variable"]
        prevalence_variable = target_def["prevalence_variable"]

        baseline_assessment.loc[:, "Event_within_10y"] = (
            (
                baseline_assessment[time_target_variable] > baseline_assessment["TIME"]
            )  # Time of event after baseline (equal means prevalent)
            & (
                baseline_assessment[time_target_variable]
                <= baseline_assessment["TIME"] + 10 * 365.25
            )  # Time of event within 10 years
            & (baseline_assessment[target_variable] == 1)  # Event = 1
        ).astype(bool)

        baseline_assessment.loc[:, "Prevalent"] = (
            (baseline_assessment[time_target_variable] < baseline_assessment["TIME"])
            & (baseline_assessment[target_variable] == 1)
        ) | (baseline_assessment[prevalence_variable])

        subset = baseline_assessment
        subset = subset.loc[~mask]
        subset = subset[~subset["Prevalent"]]

        y_true = subset["Event_within_10y"]

        prompt_parts_subset = prompt_parts.loc[subset.index]
        tab_df_subset = tab_df.loc[subset.index]

        # different prompt versions
        if unit == "mgdl":
            inputs = [
                "SEX",
                "AGE",
                "TOTCHOL",
                "HDLC",
                "SYSBP",
                "CURSMOKE",
                "DIABETES",
                "BPMEDS",
            ]
        elif unit == "mmoll":
            inputs = [
                "SEX",
                "AGE",
                "TOTCHOL_2",
                "HDLC_2",
                "SYSBP",
                "CURSMOKE",
                "DIABETES",
                "BPMEDS",
            ]
        prompts = (
            prompt_parts_subset[inputs]
            .apply(lambda x: ";\n".join(x), axis=1)
            .rename("prompt")
            .to_frame()
        )
        prompts["completion"] = y_true.map({False: "Negative", True: "Positive"})
        name = f"prompt_parts_{unit}_{target_name}.parquet"
        prompts.to_parquet(ukb_data_utils.ASSETS_PATH / "framingham" / name)

        # tabular data
        tab_df_subset["Prevalent"] = subset["Prevalent"]
        tab_df_subset["target"] = y_true
        tab_df_out = tab_df_subset[inputs + ["target"]].copy()
        tab_df_out.to_parquet(
            ukb_data_utils.ASSETS_PATH
            / "framingham"
            / f"tabular_{unit}_{target_name}.parquet"
        )
