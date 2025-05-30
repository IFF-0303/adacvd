import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from IPython.display import display

from exploration.frederike.visualization import color_palette
from exploration.frederike.visualization.visualization_utils import setup_plotting
from pandora.data import ukb_data_utils, ukb_field_ids
from pandora.data.ukb_data_utils import ASSETS_PATH

textwidth = 1.2 * 6.52  # inches

n = 12
context = "paper"
plt.colormaps.unregister("custom_palette")
sns.set_theme(style="whitegrid")

sns.set_context(context, font_scale=0.9)
# plt.rcParams["font.family"] = "Arial"  # "Arial"
# plt.rcParams["font.size"] = 4
plt.rcParams["xtick.bottom"] = True

plt.rcParams["axes.labelsize"] = plt.rcParams["xtick.labelsize"]
plt.rcParams["axes.titlesize"] = plt.rcParams["xtick.labelsize"]


pd.options.mode.chained_assignment = None  # default='warn'


df = pd.read_csv(ukb_data_utils.ASSETS_PATH / "framingham/frmgham2.csv")

# UTILS
df["MI_STRK"] = df["MI_FCHD"] | df["STROKE"]
df["TIME_MI_STRK"] = df[["TIMEMIFC", "TIMESTRK"]].min(axis=1)
df["PREVCVD"] = (
    df["PREVCHD"] | df["PREVMI"] | df["PREVSTRK"]
)  # df["PREVAP"] # TODO: include AP and coronary heart disease or not?

df["PREV_STRK_MI"] = df["PREVMI"] | df["PREVSTRK"]
df["CVD_STRK_MI"] = df["CVD"] | df["STROKE"] | df["MI_FCHD"]
df["TIME_CVD_STRK_MI"] = df[["TIMECVD", "TIMESTRK", "TIMEMIFC"]].min(axis=1)

# df[df["MI_FCHD"] == 1]["CVD_STRK"].value_counts()

baseline_assessment = df[df["PERIOD"] == 3].set_index("RANDID")
baseline_assessment["TIMEPOINT_PERIOD_3"] = baseline_assessment.groupby("RANDID")[
    "TIME"
].min()  # not necessary, there is only one row anyways


# DESCRIPTIVE STATS
# number of participants
n_participants = len(df["RANDID"].unique())
print(f"Number of participants: {n_participants}")

# number of visits per participant
print("Each participant took place in 1-3 visits (called 'PERIOD').")
display(df.groupby("RANDID")["PERIOD"].count().value_counts().sort_index())

# RISK FACTORS
print("Distribution of risk factors")
display(baseline_assessment["SEX"].value_counts())
display(baseline_assessment["AGE"].describe())
display(baseline_assessment["SYSBP"].describe())
display(baseline_assessment["DIABP"].describe())
display(baseline_assessment["BPMEDS"].value_counts())
display(baseline_assessment["CURSMOKE"].value_counts())
display(baseline_assessment["CIGPDAY"].describe())
display(baseline_assessment["educ"].describe())
display(baseline_assessment["TOTCHOL"].describe())
display(baseline_assessment["HDLC"].describe())
display(baseline_assessment["BMI"].describe())
display(baseline_assessment["GLUCOSE"].describe())
display(baseline_assessment["DIABETES"].value_counts())
display(baseline_assessment["HEARTRTE"].describe())

# OUTCOMES PREVALENT AT BASELINE
print("Distribution of outcomes prevalent at baseline")
display(baseline_assessment["PREVAP"].value_counts(normalize=True))
display(baseline_assessment["PREVCHD"].value_counts(normalize=True))
display(baseline_assessment["PREVMI"].value_counts(normalize=True))
display(baseline_assessment["PREVSTRK"].value_counts(normalize=True))
display(baseline_assessment["PREVHYP"].value_counts(normalize=True))
display(baseline_assessment["PREVCVD"].value_counts(normalize=True))
display(baseline_assessment["PREV_STRK_MI"].value_counts(normalize=True))

# OUTCOMES over entire period
# Either "CVD" or ("MI_FCHD" and "STROKE") are the outcomes of interest
display(baseline_assessment["CVD"].value_counts(normalize=True))
display(baseline_assessment["MI_STRK"].value_counts(normalize=True))
display(baseline_assessment["MI_FCHD"].value_counts(normalize=True))
display(baseline_assessment["STROKE"].value_counts(normalize=True))

display(baseline_assessment["CVD_STRK_MI"].value_counts(normalize=True))

# OUTCOMES over 10 year period
# If we are not looking at period 1, we need to adjust this
# e.g. Period 3 + 10 years
# Time of Event < Time Period 3 --> Prevalent
# Time Period 3 <= Time of Event <= Time Period 3 + 10 years --> Incident
# Time of Event > Time Period 3 + 10 years --> Censored

target_variable = "CVD_STRK_MI"
time_target_variable = "TIME_CVD_STRK_MI"
prevalence_variable = "PREV_STRK_MI"  # we may need to include CHD as well?

target_variable = "MI_STRK"
time_target_variable = "TIME_MI_STRK"
prevalence_variable = "PREV_STRK_MI"

target_variable = "CVD"
time_target_variable = "TIMECVD"
prevalence_variable = "PREV_STRK_MI"

# special case: time of event at baseline

baseline_assessment["Event_within_10y"] = (
    (
        baseline_assessment[time_target_variable] > baseline_assessment["TIME"]
    )  # Time of event after baseline (equal means prevalent)
    & (
        baseline_assessment[time_target_variable]
        <= baseline_assessment["TIME"] + 10 * 365.25
    )  # Time of event within 10 years
    & (baseline_assessment[target_variable] == 1)  # Event = 1
).astype(bool)

baseline_assessment["Prevalent"] = (
    (
        (baseline_assessment[time_target_variable] < baseline_assessment["TIME"])
        & (baseline_assessment[target_variable] == 1)
    )
) | (baseline_assessment[prevalence_variable])


# a = baseline_assessment[
#     (baseline_assessment[time_target_variable] < baseline_assessment["TIME"])
#     & (baseline_assessment[target_variable] == 1)
#     & (baseline_assessment[prevalence_variable] == 0)
# ]


# check consistency with incidents
assert (
    baseline_assessment[baseline_assessment["Event_within_10y"]][prevalence_variable]
    == 0
).all(), "Prevalent cases should not have incidents"

# check consistency with overall period
assert (
    baseline_assessment["Event_within_10y"].sum()
    <= baseline_assessment[target_variable].sum()
).all(), "Incidents should be less or equal to overall cases"

print(f"Target variable: {target_variable}")
print(f"Time of target variable: {time_target_variable}")
print(f"Prevalence variable: {prevalence_variable}")

print("Distribution of target variable within 10 years")
display(baseline_assessment["Event_within_10y"].value_counts(normalize=True))

print("Prevalence at baseline")
display(baseline_assessment["Prevalent"].value_counts(normalize=True))

print("Distribution of target variable within 10 years without prevalence at baseline")
display(
    baseline_assessment[baseline_assessment["Prevalent"] == False][
        "Event_within_10y"
    ].value_counts(normalize=True)
)


# CENSORING TIMES
subset = baseline_assessment[baseline_assessment["Prevalent"] == False]
subset["TIME_FROM_BASELINE"] = subset[time_target_variable] - subset["TIME"]
sns.displot(subset, x="TIME_FROM_BASELINE", kind="ecdf", hue=target_variable)
plt.vlines(10 * 365.25, 0, 1, color="gray", linestyle="--", label="10 years")
plt.legend()
plt.show()

# use precomputed csv

df_fram = pd.read_parquet(
    ukb_data_utils.ASSETS_PATH / "framingham/tabular_mgdl_ANYCHD.parquet"
)

# create path "figs"
from pathlib import Path

Path("figs").mkdir(parents=True, exist_ok=True)

df_fram["Event_within_10y"] = df_fram["target"].copy()

inputs = df_fram.copy()
# inputs["UNTREATED_SYSBP"] = inputs["SYSBP"].copy()
# inputs["TREATED_SYSBP"] = inputs["SYSBP"].copy()
# inputs.loc[inputs["BPMEDS"] == 0, "TREATED_SYSBP"] = None
# inputs.loc[inputs["BPMEDS"] == 1, "UNTREATED_SYSBP"] = None

inputs["CURSMOKE"] = inputs["CURSMOKE"].map({"Never": 0, "Current": 1})
inputs["DIABETES"] = inputs["DIABETES"].map({"No": 0, "Yes": 1})
inputs["SEX"] = inputs["SEX"].map({"Female": "F", "Male": "M"})


# Compare main risk factors with UKB

df_c3_10y = subset

patients_to_remove = pd.read_csv(
    "~/biobank/biobank-llm/w60520_20241217.csv", header=None
)
patients_to_remove = patients_to_remove[0].values.tolist()

filename = "MACE_ADO_EXTENDED_no_previous_target"
with open(ASSETS_PATH / f"subsets/ukb_2024_02/{filename}.json", "r") as f:
    subset = yaml.load(f, Loader=yaml.CLoader)

df_ukb = pd.read_csv(
    ukb_data_utils.ASSETS_PATH / "risk_scores" / "risk_score_inputs.csv",
).reset_index()

len(df_ukb)

df_ukb["to_remove"] = df_ukb["eid"].isin(patients_to_remove)
df_ukb = df_ukb[~df_ukb["to_remove"]]


len(df_ukb)

df_ukb["subset"] = df_ukb["eid"].isin(subset)
df_ukb = df_ukb[df_ukb["subset"]]
df_ukb = df_ukb.set_index("eid")

ukb_target = pd.read_parquet(
    ukb_data_utils.ASSETS_PATH / "targets" / "ukb_2024_02" / "targets.parquet"
)
target_name = "MACE_ADO_EXTENDED_10y"

# Compare distributions of risk factors

# df_comp = pd.DataFrame(
#     index=df_c3_10y.index.values.tolist() + df_ukb.index.values.tolist()
# )

df_comp = pd.DataFrame(
    index=df_fram.index.values.tolist() + df_ukb.index.values.tolist()
)
df_c3_10y = inputs.copy()

df_comp["source"] = None
df_comp.loc[df_c3_10y.index, "source"] = "Framingham"
df_comp.loc[df_ukb.index, "source"] = "UKB"

df_comp["target"] = None
df_comp.loc[df_c3_10y.index, "target"] = df_c3_10y["Event_within_10y"]
df_comp.loc[df_ukb.index, "target"] = ukb_target[target_name]

print(len(df_comp))
df_comp = df_comp[~df_comp.index.duplicated(keep="first")]
print(len(df_comp))

RISK_FACTORS = {
    "gender": "SEX",
    "age": "AGE",
    # "bmi": "BMI",
    "smoker": "CURSMOKE",
    "total_cholesterol": "TOTCHOL",
    "hdl_cholesterol": "HDLC",
    "systolic_bp": "SYSBP",
    "diabetic": "DIABETES",
    "BP_medication": "BPMEDS",
}

NAMES = {
    "gender": "Gender",
    "age": "Age",
    # "bmi": "BMI",
    "smoker": "Smoking Status",
    "total_cholesterol": "Total Cholesterol",
    "hdl_cholesterol": "HDL Cholesterol",
    "systolic_bp": "Systolic Blood Pressure",
    "diabetic": "Diabetes",
    "BP_medication": "Blood Pressure Medication",
    "target": "CVD Event within 10 years",
}


palette = [color_palette.BLUE[1], color_palette.TEAL[0]]
sns.set_palette(palette)


for ukb_col, frm_col in RISK_FACTORS.items():
    df_comp[ukb_col] = None
    df_comp.loc[df_c3_10y.index, ukb_col] = df_c3_10y[frm_col]
    df_comp.loc[df_ukb.index, ukb_col] = df_ukb[ukb_col]

    if ukb_col == "gender":
        df_comp[ukb_col] = df_comp[ukb_col].replace({1: "M", 2: "F"})
    if ukb_col in ["smoker", "diabetic"]:
        df_comp[ukb_col] = df_comp[ukb_col].astype(bool)


for col in RISK_FACTORS.keys():
    if col in ["gender", "smoker", "diabetic", "BP_medication"]:
        continue
    else:
        sns.displot(
            df_comp,
            x=col,
            hue="source",
            kind="kde",
            common_norm=False,
            height=4,
            aspect=1,
        )
        plt.xlabel(NAMES[col])
        plt.savefig(
            f"figs/comparison_framingham_ukb_{col}.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.show()

# sns.countplot(data=df_comp, x=col, hue="source")


# Compare distributions of binary risk factors
for col in ["gender", "smoker", "diabetic", "BP_medication", "target"]:
    (
        df_comp.groupby([col, "source"]).size() / df_comp.groupby(["source"]).size()
    ).unstack().plot(kind="bar", figsize=(4, 4))

    plt.xlabel(NAMES[col])
    plt.savefig(
        f"figs/comparison_framingham_ukb_{col}.pdf", bbox_inches="tight", format="pdf"
    )

# for col in RISK_FACTORS.keys():
#     if col in ["gender", "smoker", "diabetic", "BP_medication"]:
#         tmp_df = (
#             (
#                 df_comp.groupby(["source", col, "target"]).size()
#                 / df_comp.groupby(["source", col]).size()
#             )
#             .to_frame("percentage")
#             .reset_index()
#         )
#         sns.catplot(
#             x=col,
#             y="percentage",
#             hue="target",
#             col="source",
#             data=tmp_df,
#             kind="bar",
#             dodge=True,
#         )
#         plt.suptitle(
#             f"Comparison of distribution of risk score input {col} and CVD outcome",
#             y=1.05,
#         )
#         # plt.savefig(
#         #     f"figs/comparison_framingham_ukb_target_{col}.pdf",
#         #     bbox_inches="tight",
#         #     format="pdf",
#         # )
#         plt.show()
#     else:
#         sns.displot(
#             data=df_comp,
#             x=col,
#             col="source",
#             kind="kde",
#             hue="target",
#             common_norm=False,
#             aspect=1.5,
#             height=4,
#         )
#         # title for the entire plot
#         plt.suptitle(
#             f"Comparison of distribution of risk score input {col} and CVD outcome",
#             y=1.05,
#         )
#         # plt.savefig(
#         #     f"figs/comparison_framingham_ukb_target_{col}.pdf",
#         #     bbox_inches="tight",
#         #     format="pdf",
#         # )
#         plt.show()
