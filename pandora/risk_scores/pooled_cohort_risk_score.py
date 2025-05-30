import math
from dataclasses import dataclass
from typing import Literal

from pandora.data.ukb_data_utils import isNaN


# COEFFICIENTS
@dataclass(frozen=True)
class Coefficients:
    ln_age: float
    ln_age_sq: float
    ln_total_cholesterol: float
    ln_age_times_ln_total_cholesterol: float
    ln_hdl_cholesterol: float
    ln_age_times_ln_hdl_cholesterol: float
    ln_treated_systolic_bp: float
    ln_age_times_treated_systolic_bp: float
    ln_untreated_systolic_bp: float
    ln_age_times_untreated_systolic_bp: float
    smoker: bool
    ln_age_times_smoker: float
    diabetic: bool
    baseline_survival: float
    mean_sum_coefficient_times_value: float


# Coefficients from Goff et al. (2013) Appendix Table A
COEFFICIENTS_WHITE_FEMALE = Coefficients(
    ln_age=-29.799,
    ln_age_sq=4.884,
    ln_total_cholesterol=13.540,
    ln_age_times_ln_total_cholesterol=-3.114,
    ln_hdl_cholesterol=-13.578,
    ln_age_times_ln_hdl_cholesterol=3.149,
    ln_treated_systolic_bp=2.019,
    ln_age_times_treated_systolic_bp=0,
    ln_untreated_systolic_bp=1.957,
    ln_age_times_untreated_systolic_bp=0,
    smoker=7.574,
    ln_age_times_smoker=-1.665,
    diabetic=0.661,
    baseline_survival=0.9665,
    mean_sum_coefficient_times_value=-29.18,
)

COEFFICIENTS_AFRICANAMERICAN_FEMALE = Coefficients(
    ln_age=17.114,
    ln_age_sq=0,
    ln_total_cholesterol=0.940,
    ln_age_times_ln_total_cholesterol=0,
    ln_hdl_cholesterol=-18.920,
    ln_age_times_ln_hdl_cholesterol=4.475,
    ln_treated_systolic_bp=29.291,
    ln_age_times_treated_systolic_bp=-6.432,
    ln_untreated_systolic_bp=27.820,
    ln_age_times_untreated_systolic_bp=-6.087,
    smoker=0.691,
    ln_age_times_smoker=0,
    diabetic=0.874,
    baseline_survival=0.9533,
    mean_sum_coefficient_times_value=86.61,
)

COEFFICIENTS_WHITE_MALE = Coefficients(
    ln_age=12.344,
    ln_age_sq=0,
    ln_total_cholesterol=11.853,
    ln_age_times_ln_total_cholesterol=-2.664,
    ln_hdl_cholesterol=-7.990,
    ln_age_times_ln_hdl_cholesterol=1.769,
    ln_treated_systolic_bp=1.797,
    ln_age_times_treated_systolic_bp=0,
    ln_untreated_systolic_bp=1.764,
    ln_age_times_untreated_systolic_bp=0,
    smoker=7.837,
    ln_age_times_smoker=-1.795,
    diabetic=0.658,
    baseline_survival=0.9144,
    mean_sum_coefficient_times_value=61.18,
)


COEFFICIENTS_AFRICANAMERICAN_MALE = Coefficients(
    ln_age=2.469,
    ln_age_sq=0,
    ln_total_cholesterol=0.302,
    ln_age_times_ln_total_cholesterol=0,
    ln_hdl_cholesterol=-0.307,
    ln_age_times_ln_hdl_cholesterol=0,
    ln_treated_systolic_bp=1.916,
    ln_age_times_treated_systolic_bp=0,
    ln_untreated_systolic_bp=1.809,
    ln_age_times_untreated_systolic_bp=0,
    smoker=0.549,
    ln_age_times_smoker=0,
    diabetic=0.645,
    baseline_survival=0.8954,
    mean_sum_coefficient_times_value=19.54,
)


# ALGORITHM
def calculate_risk(
    gender,
    race,
    age,
    total_cholesterol,
    hdl_cholesterol,
    treated_systolic_bp,
    untreated_systolic_bp,
    smoker,
    diabetic,
):
    if gender == "F":
        if race == "white":
            coefficients = COEFFICIENTS_WHITE_FEMALE
        elif race == "african_american":
            coefficients = COEFFICIENTS_AFRICANAMERICAN_FEMALE
    if gender == "M":
        if race == "white":
            coefficients = COEFFICIENTS_WHITE_MALE
        elif race == "african_american":
            coefficients = COEFFICIENTS_AFRICANAMERICAN_MALE

    # inputs
    ln_age = math.log(age)
    ln_age_sq = math.log(age) ** 2
    ln_total_cholesterol = math.log(total_cholesterol)
    ln_age_times_ln_total_cholesterol = math.log(age) * math.log(total_cholesterol)
    ln_hdl_cholesterol = math.log(hdl_cholesterol)
    ln_age_times_ln_hdl_cholesterol = math.log(age) * math.log(hdl_cholesterol)

    use_treated_systolic_bp = (
        True
        if (treated_systolic_bp is not None and isNaN(treated_systolic_bp) is False)
        else False
    )
    if use_treated_systolic_bp:
        ln_treated_systolic_bp = math.log(treated_systolic_bp)
        ln_age_times_treated_systolic_bp = math.log(age) * math.log(treated_systolic_bp)
    else:
        ln_untreated_systolic_bp = math.log(untreated_systolic_bp)
        ln_age_times_untreated_systolic_bp = math.log(age) * math.log(
            untreated_systolic_bp
        )
    smoker = int(smoker)
    ln_age_times_smoker = math.log(age) * smoker
    diabetic = int(diabetic)

    # sum of coefficient * value
    sum_coefficient_times_value = (
        coefficients.ln_age * ln_age
        + coefficients.ln_age_sq * ln_age_sq
        + coefficients.ln_total_cholesterol * ln_total_cholesterol
        + coefficients.ln_age_times_ln_total_cholesterol
        * ln_age_times_ln_total_cholesterol
        + coefficients.ln_hdl_cholesterol * ln_hdl_cholesterol
        + coefficients.ln_age_times_ln_hdl_cholesterol * ln_age_times_ln_hdl_cholesterol
        + (
            (
                coefficients.ln_treated_systolic_bp * ln_treated_systolic_bp
                + coefficients.ln_age_times_treated_systolic_bp
                * ln_age_times_treated_systolic_bp
            )
            if use_treated_systolic_bp
            else (
                +coefficients.ln_untreated_systolic_bp * ln_untreated_systolic_bp
                + coefficients.ln_age_times_untreated_systolic_bp
                * ln_age_times_untreated_systolic_bp
            )
        )
        + coefficients.ln_age_times_smoker * ln_age_times_smoker
        + coefficients.diabetic * diabetic
    )

    risk_score = 1 - coefficients.baseline_survival ** math.exp(
        sum_coefficient_times_value - coefficients.mean_sum_coefficient_times_value
    )

    return risk_score


@dataclass(frozen=True)
class Inputs:
    gender: Literal["F", "M"]
    race: Literal["white", "african_american"]
    age: int
    total_cholesterol: float  # mg/dL
    hdl_cholesterol: float  # mg/dL
    treated_systolic_bp: float  # mmHg
    untreated_systolic_bp: float  # mmHg
    smoker: bool
    diabetic: bool
