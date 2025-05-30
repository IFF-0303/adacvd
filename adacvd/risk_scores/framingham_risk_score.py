import math
from dataclasses import dataclass
from typing import Literal

from adacvd.data.ukb_data_utils import isNaN


# COEFFICIENTS
@dataclass(frozen=True)
class Coefficients:
    ln_age: float
    ln_total_cholesterol: float
    ln_hdl_cholesterol: float
    ln_treated_systolic_bp: float
    ln_untreated_systolic_bp: float
    smoker: bool
    diabetic: bool
    baseline_survival: float
    mean_sum_coefficient_times_value: float


# Coefficients from 2017 paper, D’Agostino et al., Table 2
COEFFICIENTS_FEMALE = Coefficients(
    ln_age=2.32888,
    ln_total_cholesterol=1.20904,
    ln_hdl_cholesterol=-0.70833,
    ln_untreated_systolic_bp=2.76157,
    ln_treated_systolic_bp=2.82263,
    smoker=0.52873,
    diabetic=0.69154,
    baseline_survival=0.95012,
    mean_sum_coefficient_times_value=26.1931,
)

COEFFICIENTS_MALE = Coefficients(
    ln_age=3.06117,
    ln_total_cholesterol=1.12370,
    ln_hdl_cholesterol=-0.93263,
    ln_untreated_systolic_bp=1.93303,
    ln_treated_systolic_bp=1.99881,
    smoker=0.65451,
    diabetic=0.57367,
    baseline_survival=0.88936,
    mean_sum_coefficient_times_value=23.9802,
)


# ALGORITHM
def calculate_risk(
    gender,
    age,
    total_cholesterol,
    hdl_cholesterol,
    treated_systolic_bp,
    untreated_systolic_bp,
    smoker,
    diabetic,
):
    if gender == "F":
        coefficients = COEFFICIENTS_FEMALE
    elif gender == "M":
        coefficients = COEFFICIENTS_MALE
    else:
        raise ValueError(f"Gender must be 'F' or 'M'. Value was: {gender}")

    # inputs
    ln_age = math.log(age)
    ln_total_cholesterol = math.log(total_cholesterol)
    ln_hdl_cholesterol = math.log(hdl_cholesterol)

    use_treated_systolic_bp = (
        True
        if (treated_systolic_bp is not None and isNaN(treated_systolic_bp) is False)
        else False
    )
    if use_treated_systolic_bp:
        ln_treated_systolic_bp = math.log(treated_systolic_bp)
    else:
        ln_untreated_systolic_bp = math.log(untreated_systolic_bp)

    smoker = int(smoker)
    diabetic = int(diabetic)

    # sum of coefficient * value
    sum_coefficient_times_value = (
        coefficients.ln_age * ln_age
        + coefficients.ln_total_cholesterol * ln_total_cholesterol
        + coefficients.ln_hdl_cholesterol * ln_hdl_cholesterol
        + (
            (coefficients.ln_treated_systolic_bp * ln_treated_systolic_bp)
            if use_treated_systolic_bp
            else (+coefficients.ln_untreated_systolic_bp * ln_untreated_systolic_bp)
        )
        + coefficients.smoker * smoker
        + coefficients.diabetic * diabetic
    )

    risk_score = 1 - coefficients.baseline_survival ** math.exp(
        sum_coefficient_times_value - coefficients.mean_sum_coefficient_times_value
    )

    return risk_score


@dataclass(frozen=True)
class Inputs:
    gender: Literal["F", "M"]
    age: int
    total_cholesterol: float  # mg/dL
    hdl_cholesterol: float  # mg/dL
    treated_systolic_bp: float  # mmHg
    untreated_systolic_bp: float  # mmHg
    smoker: bool
    diabetic: bool
