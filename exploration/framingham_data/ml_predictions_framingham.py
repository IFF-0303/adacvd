from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from lifelines import CoxPHFitter

from pandora.data import ukb_data_utils, ukb_features, ukb_field_ids
from pandora.risk_scores import framingham_risk_score
from pandora.utils.metrics import compute_binary_classification_metrics

base_path = Path(
    "/fast/groups/hfm-users/pandora-med-box/results/2025_01_27_framingham_tests/cox"
)

model_path = base_path / "model.pkl"
preprocessor_path = base_path / "preprocessor.pkl"

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)


df = pd.read_parquet(
    ukb_data_utils.ASSETS_PATH / "framingham/tabular_mmoll_ANYCHD.parquet"
)

ukb_fram_map = {
    str(ukb_field_ids.SEX): "SEX",
    str(ukb_field_ids.AGE_AT_ASSESSMENT_CENTER): "AGE",
    str(ukb_field_ids.HDL_CHOLESTEROL): "HDLC_2",
    str(ukb_field_ids.CHOLESTEROL): "TOTCHOL_2",
    str(ukb_field_ids.SYSTOLIC_BLOOD_PRESSURE): "SYSBP",
    str(ukb_field_ids.DIABETES): "DIABETES",
    str(ukb_field_ids.SMOKING_STATUS): "CURSMOKE",
    "BP_medication": "BPMEDS",
}

df_transformed = df.rename(columns={v: k for k, v in ukb_fram_map.items()})
processed_data = preprocessor.transform(df_transformed)

if isinstance(model, lgb.Booster):
    y_pred = model.predict(processed_data)
elif isinstance(model, CoxPHFitter):
    predictions = 1 - model.predict_survival_function(processed_data).T[365 * 10]
    y_pred = predictions.values
else:
    raise ValueError("Model type not supported")


compute_framingham_score = False
if compute_framingham_score:

    df = pd.read_parquet(
        ukb_data_utils.ASSETS_PATH / "framingham/tabular_mgdl_ANYCHD.parquet"
    )

    inputs = df.copy()
    inputs["UNTREATED_SYSBP"] = inputs["SYSBP"].copy()
    inputs["TREATED_SYSBP"] = inputs["SYSBP"].copy()
    inputs.loc[inputs["BPMEDS"] == 0, "TREATED_SYSBP"] = None
    inputs.loc[inputs["BPMEDS"] == 1, "UNTREATED_SYSBP"] = None

    inputs["CURSMOKE"] = inputs["CURSMOKE"].map({"Never": 0, "Current": 1})
    inputs["DIABETES"] = inputs["DIABETES"].map({"No": 0, "Yes": 1})
    inputs["SEX"] = inputs["SEX"].map({"Female": "F", "Male": "M"})

    fram_risk_score = inputs.apply(
        lambda x: (
            framingham_risk_score.calculate_risk(
                gender=x["SEX"],
                age=x["AGE"],
                total_cholesterol=x["TOTCHOL"],
                hdl_cholesterol=x["HDLC"],
                treated_systolic_bp=x["TREATED_SYSBP"],
                untreated_systolic_bp=x["UNTREATED_SYSBP"],
                smoker=x["CURSMOKE"],
                diabetic=x["DIABETES"],
            )
        ),
        axis=1,
    )
    y_pred = fram_risk_score


# TODO: next steps: use the actual model (using exactly these inputs), save predictions

y_true = df["target"]

metrics = compute_binary_classification_metrics(y_true, y_pred)
for k, v in metrics.items():
    print(f"{k}: {v}")
