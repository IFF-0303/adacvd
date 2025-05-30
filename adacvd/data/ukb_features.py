import logging
import re
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

from adacvd.data import ukb_data_utils, ukb_field_ids


def previous_MACE(
    df: pd.DataFrame, start_dates: pd.Series, target: str = "MACE_ADO"
) -> pd.Series:
    """Whether a participant experienced a major adverse cardiovascular event before the start date.

    Args:
        df (pd.DataFrame): UKB dataframe
        start_dates (pd.Series): Series with the start date for each participant and eid as index

    Returns:
        pd.Series: pd.Series with True, False values for each participant, and eid as index.
    """
    if df.index.name == "eid":
        df = df.reset_index()

    if target == "MACE_ADO":
        extended_def = False
    elif target == "MACE_ADO_EXTENDED":
        extended_def = True
    else:
        raise NotImplementedError(f"Target {target} is not implemented.")

    ado_incidents = ukb_data_utils.get_ado_incidents(
        df, extended_def=extended_def
    ).set_index("eid")
    m = pd.merge(
        left=start_dates.to_frame("start_date"),
        right=ado_incidents,
        on="eid",
        how="left",
    )
    m["incident_before_start_date"] = m["date"] < m["start_date"]
    previous_MACE = m.groupby("eid")["incident_before_start_date"].max().to_frame()
    previous_MACE_df = pd.DataFrame(index=df["eid"]).merge(
        previous_MACE, on="eid", how="left"
    )
    assert (previous_MACE_df.index == df.eid).all()
    return previous_MACE_df.rename(
        columns={"incident_before_start_date": "previous_MACE"}
    )


def previous_MACE_by_questionnaire(
    df: pd.DataFrame, types: List[str] = ["MI", "Stroke"]
) -> pd.Series:
    VASCULAR_HEART_PROBLEMS = 6150
    CODES = []
    for t in types:
        if t == "MI":
            CODES.append(1)
        elif t == "Stroke":
            CODES.append(3)
        elif t == "Angina":
            CODES.append(2)
        else:
            raise ValueError("Type must be either 'MI', 'Stroke' or 'Angina'")
    df_long = ukb_data_utils.get_long_format(
        df.reset_index(), raw_col_prefix=VASCULAR_HEART_PROBLEMS
    )
    df_long = df_long[
        df_long["instance_array"].str.split(".").apply(lambda x: x[0] == "0")
    ]
    df_long["previous_MACE"] = df_long[VASCULAR_HEART_PROBLEMS].isin(CODES)
    previous_MACE = df_long.groupby("eid")["previous_MACE"].max().to_frame()
    previous_MACE_df = pd.DataFrame(index=df["eid"]).merge(
        previous_MACE, on="eid", how="left"
    )
    assert (previous_MACE_df.index == df.eid).all()
    return previous_MACE_df.rename(
        columns={"previous_MACE": "previous_MACE_by_questionnaire"}
    )


def highest_qualification(df: pd.DataFrame) -> pd.Series:

    # order of qualifications from lowest to highest
    qualification_order = [
        "Prefer not to answer",
        "None of the above",
        "CSEs or equivalent",
        "O levels/GCSEs or equivalent",
        "A levels/AS levels or equivalent",
        "Other professional qualifications eg: nursing, teaching",
        "NVQ or HND or HNC or equivalent",
        "College or University degree",
    ]

    def get_highest_qualification(x: str):
        highest_qualification = None
        for qual in qualification_order:
            pattern = re.compile(r"\b" + re.escape(qual) + r"\b")
            if re.search(pattern, x) is not None:
                highest_qualification = qual
        return highest_qualification if highest_qualification is not None else pd.NA

    qualifications = get_feature_values_from_field_id(
        df, ukb_field_ids.QUALIFICATIONS
    ).rename(columns={str(ukb_field_ids.QUALIFICATIONS): "Qualifications"})

    return (
        qualifications["Qualifications"]
        .apply(lambda x: get_highest_qualification(str(x)))
        .rename("highest_qualification")
    )


def medication(df: pd.DataFrame, tpye_of_medication: str) -> pd.Series:
    if df.index.name == "eid":
        df = df.reset_index()

    if tpye_of_medication.lower() == "cholesterol":
        MEDICATION_CODES_MALE = [1]  # Coding 100625
        MEDICATION_CODES_FEMALE = [1]  # Coding 100626
        name = "Cholesterol_medication"
    elif tpye_of_medication.lower() == "bp":
        MEDICATION_CODES_MALE = [2]  # Coding 100625
        MEDICATION_CODES_FEMALE = [2]  # Coding 100626
        name = "BP_medication"
    elif tpye_of_medication.lower() == "diabetes":
        MEDICATION_CODES_MALE = [3]  # Coding 100625
        MEDICATION_CODES_FEMALE = [3]  # Coding 100626
        name = "Diabetes_medication"
    else:
        raise ValueError("Medication must be either 'cholesterol' or 'BP'")

    df_long_male = ukb_data_utils.get_long_format(
        df.reset_index(),
        raw_col_prefix=ukb_field_ids.MEDICATION_CHOLESTEROL_BP_DIABETES_MALE,
    )
    df_long_female = ukb_data_utils.get_long_format(
        df.reset_index(),
        raw_col_prefix=ukb_field_ids.MEDICATION_CHOLESTEROL_BP_DIABETES_FEMALE,
    )

    # filter only instance 0
    df_long_male = df_long_male[
        df_long_male["instance_array"].str.split(".").apply(lambda x: x[0] == "0")
    ]
    df_long_female = df_long_female[
        df_long_female["instance_array"].str.split(".").apply(lambda x: x[0] == "0")
    ]

    # filter only specific medication
    df_long_male[name] = df_long_male[
        ukb_field_ids.MEDICATION_CHOLESTEROL_BP_DIABETES_MALE
    ].isin(MEDICATION_CODES_MALE)

    df_long_female[name] = df_long_female[
        ukb_field_ids.MEDICATION_CHOLESTEROL_BP_DIABETES_FEMALE
    ].isin(MEDICATION_CODES_FEMALE)

    cols = ["eid", name, "raw_col_name"]
    df_all = pd.concat(
        [
            df_long_male.loc[df_long_male[name], cols],
            df_long_female.loc[df_long_female[name], cols],
        ],
        axis=0,
    )

    result = pd.DataFrame(index=df["eid"])
    result[name] = df_all.groupby("eid")[name].max()
    result[name] = result[name].fillna(False)
    return result[name]


def BP_medication(df: pd.DataFrame) -> pd.Series:
    return medication(df, tpye_of_medication="BP")


def CHOLESTEROL_medication(df: pd.DataFrame) -> pd.Series:
    return medication(df, tpye_of_medication="Cholesterol")


def DIABETES_medication(df: pd.DataFrame) -> pd.Series:
    return medication(df, tpye_of_medication="Diabetes")


def family_history_of_CVD(df: pd.DataFrame) -> pd.Series:
    if df.index.name == "eid":
        df = df.reset_index()

    df_all = pd.DataFrame(index=df["eid"])

    # Coding 1010
    codes = [1, 2]
    relatives = {
        "Mother": ukb_field_ids.ILLNESS_MOTHER,
        "Father": ukb_field_ids.ILLNESS_FATHER,
        "Siblings": ukb_field_ids.ILLNESS_SIBLINGS,
    }

    for relative in relatives:
        col = relatives[relative]

        df_long = ukb_data_utils.get_long_format(
            df.reset_index(),
            raw_col_prefix=col,
        )

        # filter only instance 0
        df_long = df_long[
            df_long["instance_array"].str.split(".").apply(lambda x: x[0] == "0")
        ]

        # filter only specific codes
        df_long[relative] = df_long[col].isin(codes)
        df_relative = df_long.groupby("eid")[relative].max().to_frame()

        ["eid", relative]

        df_all = pd.merge(
            right=df_all,
            left=df_relative,
            how="outer",
            on="eid",
        )

    result = pd.DataFrame(index=df["eid"])
    name = "family_history_of_CVD"
    result[name] = df_all[[*relatives.keys()]].max(axis=1)
    result[name] = result[name].fillna(False)
    return result[name]


def eGFR(age: pd.Series, sex: pd.Series, creatinine: pd.Series) -> pd.Series:
    """
    Estimate glomerular filtration rate (eGFR) using the CKD-EPI race-free equation
    Args:
        age (pd.Series): age in years
        sex (pd.Series): sex, encoded as "M" or "F", or "Male" and "Female"
        creatinine (pd.Series): serum creatinine in µmol/L
    Returns:
        pd.Series: estimated GFR
    """
    female_mask = (sex == "F") | (sex == "Female")
    creatinine_boundary_mask_female = creatinine <= 61.9
    creatinine_boundary_mask_male = creatinine <= 79.6

    divisor = pd.Series(None, index=age.index)
    divisor.loc[female_mask] = 61.88
    divisor.loc[~female_mask] = 79.56

    exponent = pd.Series(None, index=age.index)
    exponent.loc[female_mask & creatinine_boundary_mask_female] = -0.241
    exponent.loc[female_mask & ~creatinine_boundary_mask_female] = -1.200
    exponent.loc[~female_mask & creatinine_boundary_mask_male] = -0.302
    exponent.loc[~female_mask & ~creatinine_boundary_mask_male] = -1.200

    factor = pd.Series(1.0, index=age.index)
    factor.loc[female_mask] = 1.012

    vals = 142 * (creatinine / divisor) ** exponent * 0.9938**age * factor

    return vals.rename("eGFR")


def eGFR_feature(df: pd.DataFrame) -> pd.Series:
    """

    Args:
        df (pd.DataFrame): raw UKB dataframe

    Returns:
        pd.Series: estimated GFR
    """
    age = get_feature_values_from_field_id(df, ukb_field_ids.AGE_AT_ASSESSMENT_CENTER)[
        str(ukb_field_ids.AGE_AT_ASSESSMENT_CENTER)
    ]
    sex = get_feature_values_from_field_id(df, ukb_field_ids.SEX)[
        str(ukb_field_ids.SEX)
    ]
    creatinine = get_feature_values_from_field_id(df, ukb_field_ids.CREATININE)[
        str(ukb_field_ids.CREATININE)
    ]
    return eGFR(age=age, sex=sex, creatinine=creatinine)


def ICD_codes(df: pd.DataFrame, start_dates: pd.Series, ICD_10=True) -> pd.Series:
    """ICD codes up to the participants start date"""

    if ICD_10:
        icd_col = ukb_field_ids.ICD_10_FIELD_ID
        icd_date_col = ukb_field_ids.ICD_10_DATE
        col_name = "ICD_10"
    else:
        icd_col = ukb_field_ids.ICD_9_FIELD_ID
        icd_date_col = ukb_field_ids.ICD_9_DATE
        col_name = "ICD_9"

    df_icd = df.filter(regex=f"^{icd_col}-")
    df_date = df.filter(regex=f"^{icd_date_col}-").apply(pd.to_datetime)
    start_dates_df = pd.concat([start_dates] * len(df_date.columns), axis=1)
    start_dates_df.columns = df_date.columns
    mask = df_date < start_dates_df
    mask.columns = df_icd.columns

    df_icd_filtered = df_icd.where(mask, inplace=False)
    vals = get_feature_values_from_field_id(df_icd_filtered, icd_col)
    return vals.rename(columns={str(icd_col): col_name})


def ICD_10_codes(df: pd.DataFrame, start_dates: pd.Series) -> pd.Series:
    return ICD_codes(df, start_dates=start_dates, ICD_10=True)


def ICD_9_codes(df: pd.DataFrame, start_dates: pd.Series) -> pd.Series:
    return ICD_codes(df, start_dates=start_dates, ICD_10=False)


FEATURES = {
    "highest_qualification": {
        "func": highest_qualification,
        "prompt": "Highest Qualification",
    },
    "BP_medication": {"func": BP_medication, "prompt": "Blood pressure medication"},
    "Cholesterol_medication": {
        "func": CHOLESTEROL_medication,
        "prompt": "Cholesterol lowering medication",
    },
    "Diabetes_medication": {
        "func": DIABETES_medication,
        "prompt": "Medication for diabetes",
    },
    "eGFR": {
        "func": eGFR_feature,
        "prompt": "Estimated glomerular filtration rate (eGFR)",
    },
    "ICD_10": {
        "func": ICD_10_codes,
        "prompt": "Diagnoses recorded in previous hospital inpatient stays (ICD-10)",
    },
    "ICD_9": {
        "func": ICD_9_codes,
        "prompt": "Diagnoses recorded in previous hospital inpatient stays (ICD-9)",
    },
}


# Feature Groups


def get_feature_names_from_feature_group(fg_name):
    with open("config/ukb_data/feature_groups/meta/feature_groups.yaml", "r") as f:
        FG_field_ids = yaml.safe_load(f)
        field_ids = [str(x) for x in FG_field_ids[fg_name].get("field_ids", [])]
        features = FG_field_ids[fg_name].get("features", [])
        return field_ids + features


# Features from Field ID


class Feature_Strategy:
    def __init__(
        self,
        instances: List[int],
        arrays: List[int],
        agg: str,
        replace_strings: bool = False,
        coding_nr: Optional[int] = None,
    ) -> None:
        self.instances = instances
        self.arrays = arrays
        self.agg = agg
        self.replace_strings = replace_strings
        self.coding_nr = coding_nr

        # TODO: missing values

    def get_values(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns
        # filter instances
        cols = [
            col
            for col in cols
            if int(col.split("-")[1].split(".")[0]) in self.instances
        ]
        if self.arrays is not None:  # "None" means all arrays
            cols = [
                col
                for col in cols
                if int(col.split("-")[1].split(".")[1]) in self.arrays
            ]

        assert len(cols) > 0, "No columns found for instances and arrays"

        if self.coding_nr is not None:
            self.coding_dict = self.get_coding_dict(dtype=df[cols].dtypes.iloc[0])
            df_ = pd.DataFrame(index=df.index, columns=cols)
            for col in cols:
                # for Ethnic Background (21000) use the top level category only (first digit)
                if col in [
                    "21000-0.0",
                    "21000-1.0",
                    "21000-2.0",
                    "21000-3.0",
                ]:
                    df[col] = df[col].apply(lambda x: float(str(x)[0]) if x > 0 else x)
                if col in ["20115-0.0", "20115-1.0", "20115-2.0"]:
                    df[col] = df[col].apply(
                        lambda x: float(str(x)[0]) * 100 if x > 0 else x
                    )

            df_[cols] = df[cols].map(lambda x: self.coding_dict.get(x, x))
            df = df_.copy()

        # replace strings
        if self.replace_strings:
            # find all strings that cannot be converted to float

            def is_digit(x):
                try:
                    float(x)
                    return True
                except ValueError:
                    return False

            for col in cols:
                special_cases = {
                    "Less than an hour a day": 0.5,
                    "Less than 1 year ago": 0.5,
                    "Less than a year ago": 0.5,
                    "Less than once a week": 0,
                    "Less than one mile": 1,
                    "Less than one a day": 0,
                }

                df.loc[:, col] = df[col].replace(special_cases)

                cals = df[col].apply(
                    lambda x: x if not is_digit(x) else pd.NA
                    # lambda x: pd.NA if not ukb_data_utils.is_float(x) else x
                )
                logging.info(f"For column {col} found strings: {cals.unique()}")

                df.loc[:, col] = df[col].apply(
                    lambda x: pd.NA if not is_digit(x) else x
                )
                df[col] = df[col].astype(pd.Float32Dtype())
                df[col] = df[col].astype(float)

        if self.agg == "mean":
            return df[cols].mean(axis=1).to_frame(cols[0].split("-")[0])
        elif self.agg == "sum":
            return df[cols].sum(axis=1).to_frame(cols[0].split("-")[0])
        elif self.agg == "max":
            return df[cols].max(axis=1).to_frame(cols[0].split("-")[0])
        elif self.agg == "min":
            return df[cols].min(axis=1).to_frame(cols[0].split("-")[0])
        elif self.agg == "all":
            vals = df[cols].apply(
                lambda x: list(x.values.astype("str")[~x.isna()]), axis=1
            )
            vals = vals.apply(lambda x: pd.NA if len(x) == 0 else x)
            return vals.to_frame(cols[0].split("-")[0])
        elif self.agg == "first":
            return df[cols].iloc[:, 0].to_frame(cols[0].split("-")[0])
        else:
            raise ValueError(f"Aggregation method {self.agg} not recognized")

    def get_coding_dict(self, dtype) -> dict:
        coding = pd.read_csv(ukb_data_utils.CODING_PATH, sep="\t")
        coding_df = coding[coding["Coding"] == int(self.coding_nr)]
        try:
            coding_df.loc[:, "Value"] = coding_df["Value"].astype(dtype)
        except ValueError:
            coding_df.loc[:, "Value"] = coding_df["Value"].astype(
                pd.api.types.infer_dtype(coding_df["Value"])
            )
        coding_dict = dict(zip(coding_df["Value"], coding_df["Meaning"]))
        return coding_dict


def get_feature_strategy_from_data_field(data_field: dict) -> dict:
    """Get feature strategy from a field ID."""

    coding_nr = (
        data_field["Coding"] if ~ukb_data_utils.isNaN(data_field["Coding"]) else None
    )
    value_type = data_field["ValueType"]
    if value_type in ["Categorical single"]:
        instances = [0]
        arrays = [0]
        agg = "first"
        replace_strings = False
    elif value_type in ["Categorical multiple"]:
        instances = [0]
        arrays = None
        agg = "all"
        replace_strings = False
    elif value_type in ["Integer", "Continuous"]:
        instances = [0]
        arrays = None
        agg = "mean" if coding_nr is None else "first"
        # TODO: if Integers or Continuous, then all strings (e.g., "Do not know" or "Prefer not to answer") should be replaced by NaN
        replace_strings = True if coding_nr is not None else False
    return Feature_Strategy(
        instances=instances,
        arrays=arrays,
        agg=agg,
        coding_nr=coding_nr,
        replace_strings=replace_strings,
    )


def get_feature_values_from_field_id(df: pd.DataFrame, field_id: int) -> pd.DataFrame:
    """Get feature values from a field ID."""

    raw_col_names = ukb_data_utils.get_all_raw_col_names_from_field_id(
        df.columns, field_id
    )
    data_field = ukb_data_utils.get_data_field(field_id, data_dict=None)

    _df = df[raw_col_names]
    strategy = get_feature_strategy_from_data_field(data_field)
    feature_values = strategy.get_values(_df)

    return feature_values


def get_period_from_baseline_assessment(
    df: pd.DataFrame, years: int = 10
) -> pd.DataFrame:
    """Get the incident period for each participant. Currently, these dates are estimated based on the year and month of birth and the age at baseline.

    Args:
        df (pd.DataFrame): Raw UKB dataframe.
        years (int): Number of years to consider for the incident period

    Returns:
        pd.DataFrame: Dataframe with the start and end date for each participant. Index=eid.
    """
    AGE_AT_BASELINE = str(ukb_field_ids.AGE_AT_ASSESSMENT_CENTER) + "-0.0"
    YEAR_OF_BIRTH = str(ukb_field_ids.YEAR_OF_BIRTH) + "-0.0"
    MONTH_OF_BIRTH = str(ukb_field_ids.MONTH_OF_BIRTH) + "-0.0"

    if df.index.name == "eid":
        df = df.reset_index()

    incident_period = pd.DataFrame(df["eid"], index=df.index)
    incident_period["start_year"] = df[YEAR_OF_BIRTH] + df[AGE_AT_BASELINE]
    incident_period["start_date"] = pd.to_datetime(
        incident_period["start_year"].astype(str)
        + "-"
        + df[MONTH_OF_BIRTH].astype(str)
        + "-01"
    )
    incident_period["end_date"] = incident_period["start_date"] + pd.DateOffset(
        years=years
    )

    return incident_period.set_index("eid")[["start_date", "end_date"]]


def retrieve_subset_no_previous_target(
    df: pd.DataFrame,
    period: pd.DataFrame,
    target: str = "MACE_ADO",
) -> pd.DataFrame:
    previous_MACE_values = previous_MACE(
        df, start_dates=period["start_date"], target=target
    )

    if target == "MACE_ADO":
        types = ["MI", "Stroke"]
    elif target == "MACE_ADO_EXTENDED":
        types = ["MI", "Stroke", "Angina"]
    else:
        raise NotImplementedError(f"Target {target} is not implemented.")

    previous_MACE_by_questionnaire_values = previous_MACE_by_questionnaire(
        df, types=types
    )

    previous_MACE_df_all = pd.concat(
        [previous_MACE_values, previous_MACE_by_questionnaire_values], axis=1
    )

    previous_MACE_values_all = previous_MACE_df_all.any(axis=1)

    return ~previous_MACE_values_all.to_frame("previous_MACE")


def ADO_MACE_target(
    df: pd.DataFrame,
    start_dates: pd.Series,
    end_dates: pd.Series,
    return_date: bool = False,
    extended_def: bool = False,
) -> pd.Series:
    """Whether a participant experienced a major adverse cardiovascular within the period between start and end date.

    Args:
        df (pd.DataFrame): Raw UKB dataframe
        start_dates (pd.Series): Start date for each participant and eid as index.
        end_dates (pd.Series): End date for each participant and eid as index.
        return_date (bool): Whether to return the date of the first incident within the period.

    Returns:
        pd.Series: pd.Series with True, False values for each participant, and eid as index.
    """

    if df.index.name == "eid":
        df = df.reset_index()

    ado_incidents = ukb_data_utils.get_ado_incidents(
        df, extended_def=extended_def
    ).set_index("eid")
    period = pd.merge(
        left=start_dates.to_frame("start_date"),
        right=end_dates.to_frame("end_date"),
        on="eid",
        how="left",
    )
    m = pd.merge(
        left=period,
        right=ado_incidents,
        on="eid",
        how="left",
    )
    m["incident_within_period"] = (m["date"] >= m["start_date"]) & (
        m["date"] <= m["end_date"]
    )
    period_MACE = m.groupby("eid")["incident_within_period"].max().to_frame()
    period_MACE_df = pd.DataFrame(index=df["eid"]).merge(
        period_MACE, on="eid", how="left"
    )

    if return_date:
        min_date = pd.to_datetime(
            m[m["incident_within_period"]].groupby("eid")["date"].min()
        )
        period_MACE_df = period_MACE_df.merge(min_date, on="eid", how="left").rename(
            columns={"date": "incident_date"}
        )
    assert (period_MACE_df.index == df.eid).all()
    return period_MACE_df.rename(columns={"incident_within_period": "target"})


def sort_features_by_feature_group(
    cols: List[str], remove_duplicates: bool = True
) -> List[str]:
    with open("config/ukb_data/feature_groups/meta/feature_groups.yaml", "r") as f:
        feature_groups = yaml.safe_load(f)
    ordered = []
    for k, v in feature_groups.items():
        ordered.extend([str(x) for x in v.get("field_ids", [])])
        ordered.extend([str(x) for x in v.get("features", [])])
    cols_ordered = [x for x in ordered if x in cols]
    if remove_duplicates:
        cols_ordered = list(dict.fromkeys(cols_ordered))
    return cols_ordered
