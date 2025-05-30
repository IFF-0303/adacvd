import time

import pandas as pd

from adacvd.data import ukb_data_utils, ukb_features, ukb_field_ids
from adacvd.risk_scores import (
    QRISK,
    SCORE,
    framingham_risk_score,
    pooled_cohort_risk_score,
    prevent_risk_score,
)

save_inputs = True

df_2024_02_ = pd.read_csv(
    f"{ukb_data_utils.ASSETS_PATH}/ukb/ukb_2024_02/ukb677731.csv",
    nrows=2,
    low_memory=False,
)

FIELD_IDS_2024_02 = [
    ukb_field_ids.SEX,
    ukb_field_ids.AGE_AT_ASSESSMENT_CENTER,
    ukb_field_ids.CHOLESTEROL,
    ukb_field_ids.HDL_CHOLESTEROL,
    ukb_field_ids.SYSTOLIC_BLOOD_PRESSURE,
    ukb_field_ids.SYSTOLIC_BLOOD_PRESSURE_MANUAL,
    ukb_field_ids.SMOKING_STATUS,
    ukb_field_ids.DIABETES,
    ukb_field_ids.ETHNIC_BACKGROUND,
    ukb_field_ids.CREATININE,
    ukb_field_ids.BMI,
    ukb_field_ids.ILLNESS_MOTHER,
    ukb_field_ids.ILLNESS_FATHER,
    ukb_field_ids.ILLNESS_SIBLINGS,
    ukb_field_ids.MEDICATION_CHOLESTEROL_BP_DIABETES_MALE,
    ukb_field_ids.MEDICATION_CHOLESTEROL_BP_DIABETES_FEMALE,
]


cols_2024_02 = ["eid"]
for field_id in FIELD_IDS_2024_02:
    cols_2024_02.extend(
        ukb_data_utils.get_all_raw_col_names_from_field_id(
            df_2024_02_.columns, field_id=field_id
        )
    )


df_2024_02 = pd.read_csv(
    f"{ukb_data_utils.ASSETS_PATH}/ukb/ukb_2024_02/ukb677731.csv",
    # nrows=1000,
    low_memory=False,
    usecols=cols_2024_02,
)


df = df_2024_02.copy()

# Load meta data
codings, data_dict, field2date = ukb_data_utils.load_ukb_meta_files()
raw2clean = ukb_data_utils.get_mapping_raw2clean(
    col_names=df.columns, only_clean_col_name=True, data_dict=data_dict
)

raw2clean_fields = ukb_data_utils.get_mapping_raw2clean(
    col_names=df.columns, only_clean_col_name=False, data_dict=data_dict
)

CHOLESTEROL_MMOLL_TO_MGDL_FACTOR = 38.67

preparation_strategies = {
    ukb_field_ids.SEX: {"use_coding": True, "map_dict": {"Male": "M", "Female": "F"}},
    ukb_field_ids.AGE_AT_ASSESSMENT_CENTER: {"use_coding": False},
    ukb_field_ids.CHOLESTEROL: {
        "use_coding": False,
        "transform": lambda x: x * CHOLESTEROL_MMOLL_TO_MGDL_FACTOR,
    },
    ukb_field_ids.HDL_CHOLESTEROL: {
        "use_coding": False,
        "transform": lambda x: x * CHOLESTEROL_MMOLL_TO_MGDL_FACTOR,
    },
    ukb_field_ids.SYSTOLIC_BLOOD_PRESSURE: {"use_coding": False},
    ukb_field_ids.SYSTOLIC_BLOOD_PRESSURE_MANUAL: {"use_coding": False},
    ukb_field_ids.SMOKING_STATUS: {
        "use_coding": True,
        "map_dict": {
            "Prefer not to answer": False,
            "Never": False,
            "Previous": False,
            "Current": True,
        },
    },
    ukb_field_ids.DIABETES: {
        "use_coding": True,
        "map_dict": {
            "Do not know": False,
            "Prefer not to answer": False,
            "No": False,
            "Yes": True,
        },
    },
    ukb_field_ids.ETHNIC_BACKGROUND: {
        "use_coding": True,
        "map_dict": {  # any other ethnic background than african_american and white is considered white in the pooled cohort risk score
            "Do not know": "white",
            "Prefer not to answer": "white",
            "White": "white",
            "British": "white",
            "Irish": "white",
            "Any other white background": "white",
            "Mixed": "white",
            "White and Black Caribbean": "white",
            "White and Black African": "white",
            "White and Asian": "white",
            "Any other mixed background": "white",
            "Asian or Asian British": "white",
            "Indian": "white",
            "Pakistani": "white",
            "Bangladeshi": "white",
            "Any other Asian background": "white",
            "Black or Black British": "african_american",
            "Caribbean": "african_american",
            "African": "african_american",
            "Any other Black background": "african_american",
            "Chinese": "white",
            "Other ethnic group": "white",
        },
    },
}


def prepare_column_for_risk_score(df: pd.DataFrame, col: str) -> pd.Series:
    vals = ukb_features.get_feature_values_from_field_id(df, col)[str(col)]
    if preparation_strategies[col].get("map_dict"):
        vals = vals.map(preparation_strategies[col]["map_dict"])
    if preparation_strategies[col].get("transform"):
        vals = vals.apply(preparation_strategies[col]["transform"])
    return vals


# prepare values for pooled cohort risk score calculation

gender_vals = prepare_column_for_risk_score(df, ukb_field_ids.SEX)
age_vals = prepare_column_for_risk_score(df, ukb_field_ids.AGE_AT_ASSESSMENT_CENTER)
cholesterol_vals = prepare_column_for_risk_score(df, ukb_field_ids.CHOLESTEROL)
hdl_cholesterol_vals = prepare_column_for_risk_score(df, ukb_field_ids.HDL_CHOLESTEROL)
untreated_systolic_bp_vals = prepare_column_for_risk_score(
    df, ukb_field_ids.SYSTOLIC_BLOOD_PRESSURE
)
untreated_systolic_bp_vals_manual = prepare_column_for_risk_score(
    df, ukb_field_ids.SYSTOLIC_BLOOD_PRESSURE_MANUAL
)
untreated_systolic_bp_vals = untreated_systolic_bp_vals.combine_first(
    untreated_systolic_bp_vals_manual
)

systolic_bp_vals = untreated_systolic_bp_vals.copy().rename("systolic_bp")

treated_systolic_bp_vals = untreated_systolic_bp_vals.copy().rename(
    "treated_systolic_bp"
)
BP_medication_vals = ukb_features.BP_medication(df)
treated_systolic_bp_vals = treated_systolic_bp_vals.where(
    BP_medication_vals, other=None, axis=0
)
untreated_systolic_bp_vals = untreated_systolic_bp_vals.where(
    ~BP_medication_vals, other=None, axis=0
)

assert (
    pd.concat([treated_systolic_bp_vals, untreated_systolic_bp_vals], axis=1)
    .isna()
    .sum(axis=1)
    >= 1
).all()

cholesterol_medication_vals = ukb_features.CHOLESTEROL_medication(df)

ethnic_background_vals = prepare_column_for_risk_score(
    df, ukb_field_ids.ETHNIC_BACKGROUND
)
smoker_vals = prepare_column_for_risk_score(df, ukb_field_ids.SMOKING_STATUS)
diabetic_vals = prepare_column_for_risk_score(df, ukb_field_ids.DIABETES)


creatinine_vals = ukb_features.get_feature_values_from_field_id(
    df, ukb_field_ids.CREATININE
)
eGFR_vals = ukb_features.eGFR(
    age_vals, gender_vals, creatinine_vals[str(ukb_field_ids.CREATININE)]
)

bmi_vals = ukb_features.get_feature_values_from_field_id(df, ukb_field_ids.BMI)[
    str(ukb_field_ids.BMI)
]

# mean imputation from UKB data for missing columns
townsend_score = pd.Series(-1.29409, index=df.index, name="townsend_score")


# first degree relatives with CVD aged less than 60
# family_history_of_premature_CVD = ukb_features.family_history_of_CVD(df).rename(
#     {"family_history_of_CVD": "family_history_of_premature_CVD"}
# ) # result: 50%, as we can not filter based on age.

family_history_of_premature_CVD = pd.Series(
    False, index=df.index, name="family_history_of_premature_CVD"
)


inputs = pd.DataFrame(
    [
        gender_vals,
        age_vals,
        cholesterol_vals,
        hdl_cholesterol_vals,
        systolic_bp_vals,
        untreated_systolic_bp_vals,
        treated_systolic_bp_vals,
        smoker_vals,
        diabetic_vals,
        ethnic_background_vals,
        BP_medication_vals,
        cholesterol_medication_vals,
        eGFR_vals,
        bmi_vals,
        townsend_score,
        family_history_of_premature_CVD,
    ]
).T.rename(
    columns={
        str(ukb_field_ids.SEX): "gender",
        str(ukb_field_ids.AGE_AT_ASSESSMENT_CENTER): "age",
        str(ukb_field_ids.CHOLESTEROL): "total_cholesterol",
        str(ukb_field_ids.HDL_CHOLESTEROL): "hdl_cholesterol",
        "systolic_bp": "systolic_bp",
        str(ukb_field_ids.SYSTOLIC_BLOOD_PRESSURE): "untreated_systolic_bp",
        "treated_systolic_bp": "treated_systolic_bp",
        str(ukb_field_ids.SMOKING_STATUS): "smoker",
        str(ukb_field_ids.DIABETES): "diabetic",
        str(ukb_field_ids.ETHNIC_BACKGROUND): "race",
        "BP_medication": "BP_medication",
        "eGFR": "eGFR",
        "Cholesterol_medication": "Cholesterol_medication",
        str(ukb_field_ids.BMI): "bmi",
        "townsend_score": "townsend_score",
        "family_history_of_premature_CVD": "family_history_of_premature_CVD",
    }
)

if save_inputs:
    inputs.to_csv(ukb_data_utils.ASSETS_PATH / "risk_scores" / "risk_score_inputs.csv")


def values_complete(inputs: pd.Series, risk_score: str) -> bool:
    complete = True
    base_cols = [
        "gender",
        "age",
        "total_cholesterol",
        "hdl_cholesterol",
        "smoker",
        "diabetic",
    ]
    if risk_score == "ACC/AHA":
        base_cols.extend(["race"])
    if inputs[base_cols].isna().sum() > 0:
        complete = False

    if inputs[["untreated_systolic_bp", "treated_systolic_bp"]].isna().sum() > 1:
        complete = False

    return complete


print("Calculate Framingham Risk Score")
# measure execution time of the below function
start_time = time.time()
# calculate risk score
fram_risk_score = inputs.apply(
    lambda x: (
        framingham_risk_score.calculate_risk(
            gender=x["gender"],
            age=x["age"],
            total_cholesterol=x["total_cholesterol"],
            hdl_cholesterol=x["hdl_cholesterol"],
            treated_systolic_bp=x["treated_systolic_bp"],
            untreated_systolic_bp=x["untreated_systolic_bp"],
            smoker=x["smoker"],
            diabetic=x["diabetic"],
        )
        if values_complete(x, risk_score="FRAMINGHAM")
        else None
    ),
    axis=1,
)
end_time = time.time()
print(
    f"Execution time: {end_time - start_time:.2f} seconds for {len(fram_risk_score)} samples"
)

print("Calculate AHA/ACC Risk Score")
# measure execution time of the below function
start_time = time.time()
# calculate risk score
PCRS_risk_score = inputs.apply(
    lambda x: (
        pooled_cohort_risk_score.calculate_risk(
            gender=x["gender"],
            race=x["race"],
            age=x["age"],
            total_cholesterol=x["total_cholesterol"],
            hdl_cholesterol=x["hdl_cholesterol"],
            treated_systolic_bp=x["treated_systolic_bp"],
            untreated_systolic_bp=x["untreated_systolic_bp"],
            smoker=x["smoker"],
            diabetic=x["diabetic"],
        )
        if values_complete(x, risk_score="ACC/AHA")
        else None
    ),
    axis=1,
)
end_time = time.time()
print(
    f"Execution time: {end_time - start_time:.2f} seconds for {len(fram_risk_score)} samples"
)

print("Calculate PREVENT Risk Score")
# measure execution time of the below function
start_time = time.time()
# calculate risk score
PREVENT_risk_score = inputs.apply(
    lambda x: (
        prevent_risk_score.calculate_risk(
            gender=x["gender"],
            age=x["age"],
            total_cholesterol=x["total_cholesterol"],
            hdl_cholesterol=x["hdl_cholesterol"],
            systolic_bp=x["systolic_bp"],
            diabetic=x["diabetic"],
            smoker=x["smoker"],
            bmi=x["bmi"],
            estimated_gfr=x["eGFR"],
            anti_hypertension_medication=x["BP_medication"],
            statin_use=x["Cholesterol_medication"],
        )
        if x[
            [
                "gender",
                "age",
                "total_cholesterol",
                "hdl_cholesterol",
                "systolic_bp",
                "diabetic",
                "smoker",
                "bmi",
                "eGFR",
                "BP_medication",
                "Cholesterol_medication",
            ]
        ]
        .isna()
        .sum()
        == 0
        else None
    ),
    axis=1,
)
end_time = time.time()
print(
    f"Execution time: {end_time - start_time:.2f} seconds for {len(fram_risk_score)} samples"
)


print("Calculate QRISK Score")
# measure execution time of the below function
start_time = time.time()
# calculate risk score
qrisk_score = inputs.apply(
    lambda x: (
        QRISK.calculate_risk(
            gender=x["gender"],
            age=x["age"],
            total_cholesterol=x["total_cholesterol"] / CHOLESTEROL_MMOLL_TO_MGDL_FACTOR,
            hdl_cholesterol=x["hdl_cholesterol"] / CHOLESTEROL_MMOLL_TO_MGDL_FACTOR,
            systolic_bp=x["systolic_bp"],
            bp_treatment=x["BP_medication"],
            smoker=x["smoker"],
            townsend_score=x["townsend_score"],
            family_history_of_premature_CVD=x["family_history_of_premature_CVD"],
            bmi=x["bmi"],
        )
        if x[
            [
                "gender",
                "age",
                "total_cholesterol",
                "hdl_cholesterol",
                "systolic_bp",
                "BP_medication",
                "smoker",
                "townsend_score",
                "family_history_of_premature_CVD",
                "bmi",
            ]
        ]
        .isna()
        .sum()
        == 0
        else None
    ),
    axis=1,
)
end_time = time.time()
print(
    f"Execution time: {end_time - start_time:.2f} seconds for {len(qrisk_score)} samples"
)


print("Calculate SCORE Score")
# measure execution time of the below function
start_time = time.time()
# calculate risk score
score_score = inputs.apply(
    lambda x: (
        SCORE.calculate_risk(
            gender=x["gender"],
            age=x["age"],
            smoker=x["smoker"],
            systolic_bp=x["systolic_bp"],
            diabetes=x["diabetic"],
            total_cholesterol=x["total_cholesterol"] / CHOLESTEROL_MMOLL_TO_MGDL_FACTOR,
            hdl_cholesterol=x["hdl_cholesterol"] / CHOLESTEROL_MMOLL_TO_MGDL_FACTOR,
        )
        if x[
            [
                "gender",
                "age",
                "smoker",
                "systolic_bp",
                "diabetic",
                "total_cholesterol",
                "hdl_cholesterol",
            ]
        ]
        .isna()
        .sum()
        == 0
        else None
    ),
    axis=1,
)
end_time = time.time()
print(
    f"Execution time: {end_time - start_time:.2f} seconds for {len(score_score)} samples"
)

# save results
PATH = ukb_data_utils.ASSETS_PATH / "risk_scores"
PATH.mkdir(parents=True, exist_ok=True)


# save only when risk scores are computed on full set
if (
    len(fram_risk_score) == 502204
    and len(PCRS_risk_score) == 502204
    and len(PREVENT_risk_score) == 502204
    and len(qrisk_score) == 502204
    and len(score_score) == 502204
):
    # archive old files into archive folder
    archive_path = PATH / "archive"
    archive_path.mkdir(parents=True, exist_ok=True)

    for file in [
        "framingham_risk_scores.csv",
        "pooled_cohort_risk_scores.csv",
        "prevent_risk_scores.csv",
        "qrisk_scores.csv",
        "score_scores.csv",
    ]:
        if (PATH / file).exists():
            (PATH / file).rename(archive_path / file)

    fram_risk_score.to_csv(PATH / "framingham_risk_scores.csv")
    PCRS_risk_score.to_csv(PATH / "pooled_cohort_risk_scores.csv")
    PREVENT_risk_score.to_csv(PATH / "prevent_risk_scores.csv")
    qrisk_score.to_csv(PATH / "qrisk_scores.csv")
    score_score.to_csv(PATH / "score_scores.csv")

    print("Saved all risk scores.")
