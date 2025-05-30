# %cd '/home/fluebeck/biobank/biobank-llm'
import pandas as pd
import yaml

from pandora.data import ukb_data_utils
from pandora.data.ukb_data_utils import ASSETS_PATH
from pandora.training.dataset import get_column_names, load_split

patients_to_remove = pd.read_csv(
    "~/biobank/biobank-llm/w60520_20241217.csv", header=None
)
patients_to_remove = patients_to_remove[0].values.tolist()

filename = "MACE_ADO_EXTENDED_no_previous_target"
with open(ASSETS_PATH / f"subsets/ukb_2024_02/{filename}.json", "r") as f:
    subset = yaml.load(f, Loader=yaml.CLoader)

df_ukb = pd.read_csv(
    ukb_data_utils.ASSETS_PATH / "risk_scores" / "risk_score_inputs.csv",
)

len(df_ukb)

df_ukb["to_remove"] = df_ukb["eid"].isin(patients_to_remove)
df_ukb = df_ukb[~df_ukb["to_remove"]]

len(df_ukb)

df_ukb["subset"] = df_ukb["eid"].isin(subset)
df_ukb = df_ukb[df_ukb["subset"]]

len(df_ukb)

data_config = {"split": "config/splits/split_2025_02_17_test_100000.json"}

split = load_split(data_config)
df_ukb["train"] = df_ukb["eid"].isin(split["train"])
df_ukb["test"] = df_ukb["eid"].isin(split["test"])
df_ukb["validation"] = df_ukb["eid"].isin(split["validation"])

df_ukb[df_ukb["train"]].describe()

print(df_ukb.shape)

df_ukb = df_ukb[df_ukb["train"] | df_ukb["test"] | df_ukb["validation"]]

print(df_ukb.shape)
summary = df_ukb[df_ukb["train"]].groupby("gender").describe()

table = {}

for gender in ["F", "M"]:
    table[gender] = {}

    for col in [
        "age",
        "bmi",
        "total_cholesterol",
        "hdl_cholesterol",
        "systolic_bp",
        "eGFR",
    ]:
        median = df_ukb[df_ukb["gender"] == gender][col].median()
        std = df_ukb[df_ukb["gender"] == gender][col].std()
        table[gender][col] = f"{median:.2f} ({std:.2f})"

    for col in ["smoker", "diabetic", "BP_medication"]:
        value_true = df_ukb[df_ukb["gender"] == gender][col].mean() * 100
        table[gender][col] = f"{value_true:.2f}%"

    table[gender]["n"] = df_ukb[df_ukb["gender"] == gender].shape[0]

pd.DataFrame(table).to_csv("ukb_cohort_summary.csv")
pd.DataFrame(table)
