import math
from dataclasses import dataclass
from typing import Literal

"""
Coefficients from Khan et al. (2024), Supplementary Material S12A Base 10yr
"""


@dataclass(frozen=True)
class Coefficients:
    age_per_10_years: float
    non_hdl_c_per_1_mmol_l: float
    hdl_c_per_0_3_mmol_l: float
    sbp_lt_110_per_20_mmhg: float
    sbp_ge_110_per_20_mmhg: float
    diabetes: float
    current_smoking: float
    bmi_lt_30_per_5_kg_m2: float
    bmi_ge_30_per_5_kg_m2: float
    egfr_lt_60_per_minus_15_ml: float
    egfr_ge_60_per_minus_15_ml: float
    anti_hypertensive_use: float
    statin_use: float
    treated_sbp_ge_110_mm_hg_per_20_mm_hg: float
    treated_non_hdl_c: float
    age_per_10_years_times_non_hdl_c_per_1_mmol_l: float
    age_per_10_years_times_hdl_c_per_0_3_mmol_l: float
    age_per_10_years_times_sbp_ge_110_mm_hg_per_20_mmhg: float
    age_per_10_years_times_diabetes: float
    age_per_10_years_times_current_smoking: float
    age_per_10_years_times_bmi_ge_30_per_5_kg_m2: float
    age_per_10_years_times_egfr_lt_60_per_minus_15_ml: float
    constant: float


COEFFICIENTS_FEMALE = Coefficients(
    age_per_10_years=0.7939329,
    non_hdl_c_per_1_mmol_l=0.0305239,
    hdl_c_per_0_3_mmol_l=-0.1606857,
    sbp_lt_110_per_20_mmhg=-0.2394003,
    sbp_ge_110_per_20_mmhg=0.3600781,
    diabetes=0.8667604,
    current_smoking=0.5360739,
    bmi_lt_30_per_5_kg_m2=0,
    bmi_ge_30_per_5_kg_m2=0,
    egfr_lt_60_per_minus_15_ml=0.6045917,
    egfr_ge_60_per_minus_15_ml=0.0433769,
    anti_hypertensive_use=0.3151672,
    statin_use=-0.1477655,
    treated_sbp_ge_110_mm_hg_per_20_mm_hg=-0.0663612,
    treated_non_hdl_c=0.1197879,
    age_per_10_years_times_non_hdl_c_per_1_mmol_l=-0.0819715,
    age_per_10_years_times_hdl_c_per_0_3_mmol_l=0.0306769,
    age_per_10_years_times_sbp_ge_110_mm_hg_per_20_mmhg=-0.0946348,
    age_per_10_years_times_diabetes=-0.27057,
    age_per_10_years_times_current_smoking=-0.078715,
    age_per_10_years_times_bmi_ge_30_per_5_kg_m2=0,
    age_per_10_years_times_egfr_lt_60_per_minus_15_ml=-0.1637806,
    constant=-3.307728,
)

COEFFICIENTS_MALE = Coefficients(
    age_per_10_years=0.7688528,
    non_hdl_c_per_1_mmol_l=0.0736174,
    hdl_c_per_0_3_mmol_l=-0.0954431,
    sbp_lt_110_per_20_mmhg=-0.4347345,
    sbp_ge_110_per_20_mmhg=0.3362658,
    diabetes=0.7692857,
    current_smoking=0.4386871,
    bmi_lt_30_per_5_kg_m2=0,
    bmi_ge_30_per_5_kg_m2=0,
    egfr_lt_60_per_minus_15_ml=0.5378979,
    egfr_ge_60_per_minus_15_ml=0.0164827,
    anti_hypertensive_use=0.288879,
    statin_use=-0.1337349,
    treated_sbp_ge_110_mm_hg_per_20_mm_hg=-0.0475924,
    treated_non_hdl_c=0.150273,
    age_per_10_years_times_non_hdl_c_per_1_mmol_l=-0.0517874,
    age_per_10_years_times_hdl_c_per_0_3_mmol_l=0.0191169,
    age_per_10_years_times_sbp_ge_110_mm_hg_per_20_mmhg=-0.1049477,
    age_per_10_years_times_diabetes=-0.2251948,
    age_per_10_years_times_current_smoking=-0.0895067,
    age_per_10_years_times_bmi_ge_30_per_5_kg_m2=0,
    age_per_10_years_times_egfr_lt_60_per_minus_15_ml=-0.1543702,
    constant=-3.031168,
)


@dataclass(frozen=True)
class Inputs:
    gender: Literal["F", "M"]
    age: int
    total_cholesterol: float  # mg/dL
    hdl_cholesterol: float  # mg/dL
    systolic_bp: float  # mmHg
    diabetic: bool
    smoker: bool
    bmi: float  # kg/m2
    estimated_gfr: float  # ml/min/1.73m2
    anti_hypertension_medication: bool
    statin_use: bool


def calculate_risk(
    gender,
    age,
    total_cholesterol,
    hdl_cholesterol,
    systolic_bp,
    diabetic,
    smoker,
    bmi,
    estimated_gfr,
    anti_hypertension_medication,
    statin_use,
) -> float:
    """
    treated_sbp_ge_110_mm_hg_per_20_mm_hg
    treated_non_hdl_c
    age_per_10_years_times_non_hdl_c_per_1_mmol_l
    age_per_10_years_times_hdl_c_per_0_3_mmol_l
    age_per_10_years_times_sbp_ge_110_mm_hg_per_20_mmhg
    age_per_10_years_times_diabetes
    age_per_10_years_times_current_smoking
    age_per_10_years_times_bmi_ge_30_per_5_kg_m2
    age_per_10_years_times_egfr_lt_60_per_minus_15_ml
    constant
    """
    inputs = Inputs(
        gender=gender,
        age=age,
        total_cholesterol=total_cholesterol,
        hdl_cholesterol=hdl_cholesterol,
        systolic_bp=systolic_bp,
        diabetic=diabetic,
        smoker=smoker,
        bmi=bmi,
        estimated_gfr=estimated_gfr,
        anti_hypertension_medication=anti_hypertension_medication,
        statin_use=statin_use,
    )

    age_per_10_years = (inputs.age - 55) / 10
    non_hdl_c_per_1_mmol_l = (
        inputs.total_cholesterol - inputs.hdl_cholesterol
    ) * 0.02586 - 3.5
    hdl_c_per_0_3_mmol_l = (inputs.hdl_cholesterol * 0.02586 - 1.3) / 0.3
    sbp_lt_110_per_20_mmhg = (min(inputs.systolic_bp, 110) - 110) / 20
    sbp_ge_110_per_20_mmhg = (max(inputs.systolic_bp, 110) - 130) / 20
    diabetes = int(inputs.diabetic)
    current_smoking = int(inputs.smoker)
    bmi_lt_30_per_5_kg_m2 = (min(inputs.bmi, 30) - 25) / 5
    bmi_ge_30_per_5_kg_m2 = (max(inputs.bmi, 30) - 30) / 5
    egfr_lt_60_per_minus_15_ml = (min(inputs.estimated_gfr, 60) - 60) / -15
    egfr_ge_60_per_minus_15_ml = (max(inputs.estimated_gfr, 60) - 90) / -15
    anti_hypertensive_use = int(inputs.anti_hypertension_medication)
    statin_use = int(inputs.statin_use)
    treated_sbp_ge_110_mm_hg_per_20_mm_hg = (
        (max(inputs.systolic_bp, 110) - 130) / 20 * anti_hypertensive_use
    )
    treated_non_hdl_c = (
        (inputs.total_cholesterol - inputs.hdl_cholesterol) * 0.02586 - 3.5
    ) * statin_use
    age_per_10_years_times_non_hdl_c_per_1_mmol_l = (
        age_per_10_years * non_hdl_c_per_1_mmol_l
    )
    age_per_10_years_times_hdl_c_per_0_3_mmol_l = (
        age_per_10_years * hdl_c_per_0_3_mmol_l
    )
    age_per_10_years_times_sbp_ge_110_mm_hg_per_20_mmhg = (
        age_per_10_years * sbp_ge_110_per_20_mmhg
    )
    age_per_10_years_times_diabetes = age_per_10_years * diabetes
    age_per_10_years_times_current_smoking = age_per_10_years * current_smoking
    age_per_10_years_times_bmi_ge_30_per_5_kg_m2 = (
        age_per_10_years * bmi_ge_30_per_5_kg_m2
    )
    age_per_10_years_times_egfr_lt_60_per_minus_15_ml = (
        age_per_10_years * egfr_lt_60_per_minus_15_ml
    )
    constant = 1

    if inputs.gender == "F":
        coefficients = COEFFICIENTS_FEMALE
    elif inputs.gender == "M":
        coefficients = COEFFICIENTS_MALE

    log_odds = (
        age_per_10_years * coefficients.age_per_10_years
        + non_hdl_c_per_1_mmol_l * coefficients.non_hdl_c_per_1_mmol_l
        + hdl_c_per_0_3_mmol_l * coefficients.hdl_c_per_0_3_mmol_l
        + sbp_lt_110_per_20_mmhg * coefficients.sbp_lt_110_per_20_mmhg
        + sbp_ge_110_per_20_mmhg * coefficients.sbp_ge_110_per_20_mmhg
        + diabetes * coefficients.diabetes
        + current_smoking * coefficients.current_smoking
        + bmi_lt_30_per_5_kg_m2 * coefficients.bmi_lt_30_per_5_kg_m2
        + bmi_ge_30_per_5_kg_m2 * coefficients.bmi_ge_30_per_5_kg_m2
        + egfr_lt_60_per_minus_15_ml * coefficients.egfr_lt_60_per_minus_15_ml
        + egfr_ge_60_per_minus_15_ml * coefficients.egfr_ge_60_per_minus_15_ml
        + anti_hypertensive_use * coefficients.anti_hypertensive_use
        + statin_use * coefficients.statin_use
        + treated_sbp_ge_110_mm_hg_per_20_mm_hg
        * coefficients.treated_sbp_ge_110_mm_hg_per_20_mm_hg
        + treated_non_hdl_c * coefficients.treated_non_hdl_c
        + age_per_10_years_times_non_hdl_c_per_1_mmol_l
        * coefficients.age_per_10_years_times_non_hdl_c_per_1_mmol_l
        + age_per_10_years_times_hdl_c_per_0_3_mmol_l
        * coefficients.age_per_10_years_times_hdl_c_per_0_3_mmol_l
        + age_per_10_years_times_sbp_ge_110_mm_hg_per_20_mmhg
        * coefficients.age_per_10_years_times_sbp_ge_110_mm_hg_per_20_mmhg
        + age_per_10_years_times_diabetes * coefficients.age_per_10_years_times_diabetes
        + age_per_10_years_times_current_smoking
        * coefficients.age_per_10_years_times_current_smoking
        + age_per_10_years_times_bmi_ge_30_per_5_kg_m2
        * coefficients.age_per_10_years_times_bmi_ge_30_per_5_kg_m2
        + age_per_10_years_times_egfr_lt_60_per_minus_15_ml
        * coefficients.age_per_10_years_times_egfr_lt_60_per_minus_15_ml
        + constant * coefficients.constant
    )

    risk = math.exp(log_odds) / (1 + math.exp(log_odds))
    return risk
