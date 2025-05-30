import math
from dataclasses import dataclass
from typing import Literal

from adacvd.data.ukb_data_utils import isNaN

# Available material from Hippisley-Cox et al. (2007)
# Table 1: Incidence rates, stratified
# Table 2: Baseline characteristics for men and woman in derivation and validation cohort
# Table 3: Adjusted hazard ratios
# Table 4: Ratio of predicted to observed risk (by decile and overall)

# Notes on table 3
# continuous variables were centered


# COEFFICIENTS
@dataclass(frozen=True)
class Coefficients:
    ln_age_div_10: float
    ratio_total_cholesterol_to_HDL_cholesterol: float
    bmi: float
    family_history_of_premature_CVD: float
    smoker: float
    townsend_score: float
    systolic_bp: float
    bp_treatment: float
    systolic_bp_times_bp_treatment: float
    mean_sum_coefficient_times_value: float
    baseline_survival: float
    mean_predicted_risk: float


# Coefficients from Hippisley-Cox et al. (2007) Table 3
COEFFICIENTS_FEMALE = Coefficients(
    ln_age_div_10=87.75,
    ratio_total_cholesterol_to_HDL_cholesterol=1.001,
    bmi=1.015,
    family_history_of_premature_CVD=1.229,
    smoker=1.530,
    townsend_score=1.035,
    systolic_bp=1.005,
    bp_treatment=1.734,
    systolic_bp_times_bp_treatment=0.996,
    mean_sum_coefficient_times_value=8.107103535650033,  # computed from mean values
    baseline_survival=0.9375,  # 1 - observed risk
    mean_predicted_risk=0.0634,  # predicted risk (from Table 4)
)

COEFFICIENTS_MALE = Coefficients(
    ln_age_div_10=50.634,
    ratio_total_cholesterol_to_HDL_cholesterol=1.001,
    bmi=1.022,
    family_history_of_premature_CVD=1.300,
    smoker=1.417,
    townsend_score=1.017,
    systolic_bp=1.004,
    bp_treatment=1.847,
    systolic_bp_times_bp_treatment=0.993,
    mean_sum_coefficient_times_value=7.262825801937877,  # computed from mean values
    baseline_survival=0.9112,  # 1 - observed risk
    mean_predicted_risk=0.0886,  # predicted risk (from Table 4)
)


# ALGORITHM
def calculate_risk(
    gender,
    age,
    total_cholesterol,
    hdl_cholesterol,
    systolic_bp,  # mmHg
    bp_treatment,
    smoker,
    townsend_score,
    family_history_of_premature_CVD,
    bmi,
):
    if gender == "F":
        coefficients = COEFFICIENTS_FEMALE
    if gender == "M":
        coefficients = COEFFICIENTS_MALE

    # inputs
    ln_age_div_10 = math.log(age / 10)
    ratio_total_cholesterol_to_HDL_cholesterol = total_cholesterol / hdl_cholesterol
    smoker = int(smoker)
    bp_treatment = int(bp_treatment)
    family_history_of_premature_CVD = int(family_history_of_premature_CVD)
    systolic_bp_times_bp_treatment = systolic_bp * bp_treatment

    # sum of coefficient * value
    sum_coefficient_times_value = (
        math.log(coefficients.ln_age_div_10) * ln_age_div_10
        + math.log(coefficients.ratio_total_cholesterol_to_HDL_cholesterol)
        * ratio_total_cholesterol_to_HDL_cholesterol
        + math.log(coefficients.bmi) * bmi
        + math.log(coefficients.family_history_of_premature_CVD)
        * family_history_of_premature_CVD
        + math.log(coefficients.smoker) * smoker
        + math.log(coefficients.townsend_score) * townsend_score
        + math.log(coefficients.systolic_bp) * systolic_bp
        + math.log(coefficients.bp_treatment) * bp_treatment
        + math.log(coefficients.systolic_bp_times_bp_treatment)
        * systolic_bp_times_bp_treatment
    )

    # only for development
    # to retrieve the coefficients.mean_sum_coefficient_times_value per group (using mean imputs)
    # mean_sum_coefficient_times_value = sum_coefficient_times_value - math.log(
    #     math.log(1 - coefficients.mean_predicted_risk)
    #     / math.log(coefficients.baseline_survival)
    # )

    # print(mean_sum_coefficient_times_value)

    risk_score = 1 - coefficients.baseline_survival ** math.exp(
        sum_coefficient_times_value - coefficients.mean_sum_coefficient_times_value
    )

    return risk_score


@dataclass(frozen=True)
class Inputs:
    gender: Literal["F", "M"]
    age: int
    total_cholesterol: float
    hdl_cholesterol: float
    systolic_bp: float
    bp_treatment: bool
    smoker: bool
    townsend_score: float
    family_history_of_premature_CVD: bool
    bmi: float


if __name__ == "__main__":
    # mean values from Table 2
    mean_female = dict(
        gender="F",
        age=49,
        total_cholesterol=5.9,
        hdl_cholesterol=1.6,
        systolic_bp=132.6,
        bp_treatment=False,
        smoker=False,
        townsend_score=-1.2,
        family_history_of_premature_CVD=False,
        bmi=26,
    )

    mean_male = dict(
        gender="M",
        age=48,
        total_cholesterol=5.7,
        hdl_cholesterol=1.3,
        systolic_bp=135.7,
        bp_treatment=False,
        smoker=False,
        townsend_score=-1.1,
        family_history_of_premature_CVD=False,
        bmi=26.5,
    )

    mean_female_risk = calculate_risk(**mean_female)  # should be 0.0634 --> 0.0634
    print(mean_female_risk * 100)

    mean_male_risk = calculate_risk(**mean_male)  # should be 0.0886 --> 0.0886
    print(mean_male_risk * 100)
