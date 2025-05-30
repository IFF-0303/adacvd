import math
from dataclasses import dataclass
from typing import Literal

from adacvd.data.ukb_data_utils import isNaN


# COEFFICIENTS
@dataclass(frozen=True)
class Coefficients:
    age: float
    smoker: float
    systolic_bp: float
    diabetes: float
    total_chol: float
    hdl_chol: float
    smoker_times_age: float
    systolic_bp_times_age: float
    total_chol_times_age: float
    hdl_chol_times_age: float
    diabetes_times_age: float
    baseline_survival: float
    # mean_coefficient_times_value: float


# Coefficients from SCORE2 paper Supplementary methods Table 2 (Log SHR)
COEFFICIENTS_FEMALE = Coefficients(
    age=0.4648,
    smoker=0.7744,
    systolic_bp=0.3131,
    diabetes=0.8096,
    total_chol=0.1002,
    hdl_chol=-0.2606,
    smoker_times_age=-0.1088,
    systolic_bp_times_age=-0.0277,
    total_chol_times_age=-0.0226,
    hdl_chol_times_age=0.0613,
    diabetes_times_age=-0.1272,
    baseline_survival=0.9776,
)


COEFFICIENTS_MALE = Coefficients(
    age=0.3742,
    smoker=0.6012,
    systolic_bp=0.2777,
    diabetes=0.6457,
    total_chol=0.1458,
    hdl_chol=-0.2698,
    smoker_times_age=-0.0755,
    systolic_bp_times_age=-0.0255,
    total_chol_times_age=-0.0281,
    hdl_chol_times_age=0.0426,
    diabetes_times_age=-0.0983,
    baseline_survival=0.9605,
)


# ALGORITHM
def calculate_risk(
    gender,
    age,
    smoker,
    systolic_bp,
    diabetes,
    total_cholesterol,
    hdl_cholesterol,
):
    if gender == "F":
        coefficients = COEFFICIENTS_FEMALE
    if gender == "M":
        coefficients = COEFFICIENTS_MALE

    # inputs
    age_tr = (age - 60) / 5
    smoker = int(smoker)
    systolic_bp_tr = (systolic_bp - 120) / 20
    diabetes = int(diabetes)
    total_chol_tr = total_cholesterol - 6
    hdl_chol_tr = (hdl_cholesterol - 1.3) / 0.5
    smoker_times_age_tr = smoker * age_tr
    systolic_bp_times_age_tr = systolic_bp_tr * age_tr
    total_chol_times_age_tr = total_chol_tr * age_tr
    hdl_chol_times_age_tr = hdl_chol_tr * age_tr
    diabetes_times_age_tr = diabetes * age_tr

    # sum of coefficient * value
    sum_coefficient_times_value = (
        coefficients.age * age_tr
        + coefficients.smoker * smoker
        + coefficients.systolic_bp * systolic_bp_tr
        + coefficients.diabetes * diabetes
        + coefficients.total_chol * total_chol_tr
        + coefficients.hdl_chol * hdl_chol_tr
        + coefficients.smoker_times_age * smoker_times_age_tr
        + coefficients.systolic_bp_times_age * systolic_bp_times_age_tr
        + coefficients.total_chol_times_age * total_chol_times_age_tr
        + coefficients.hdl_chol_times_age * hdl_chol_times_age_tr
        + coefficients.diabetes_times_age * diabetes_times_age_tr
    )

    risk_score = 1 - coefficients.baseline_survival ** math.exp(
        sum_coefficient_times_value
    )

    return risk_score


@dataclass(frozen=True)
class Inputs:
    gender: Literal["F", "M"]
    age: int
    smoker: bool
    systolic_bp: float
    diabetes: bool
    total_cholesterol: float
    hdl_cholesterol: float


if __name__ == "__main__":
    # Example from Table 4
    mean_female = dict(
        gender="F",
        age=50,
        smoker=True,
        systolic_bp=140,
        diabetes=False,
        total_cholesterol=6.3,
        hdl_cholesterol=1.4,
    )

    mean_male = dict(
        gender="M",
        age=50,
        smoker=True,
        systolic_bp=140,
        diabetes=False,
        total_cholesterol=6.3,
        hdl_cholesterol=1.4,
    )

    mean_female_risk = calculate_risk(
        **mean_female
    )  # should be 0.0332 --> 0.03316289676289208

    mean_male_risk = calculate_risk(
        **mean_male
    )  # should be 0.0541 --> 0.05409837636008619
