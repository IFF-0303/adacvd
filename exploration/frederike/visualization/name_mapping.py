llm_short_names = {
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-8B",
    "microsoft/Phi-3.5-mini-instruct": "Phi-mini-3B",
    "google/gemma-2-9b-it": "Gemma-9B",
    "google/gemma-2-2b-it": "Gemma-2B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
}

feature_group_names = {
    "base_risk_score_inputs": {
        "long_name": "Framingham Risk Factors",
        "short_name": "Fram",
    },
    "additional_risk_score_inputs_aha_acc": {
        "long_name": "Ethnic Background",
        "short_name": "EB",
    },
    "additional_risk_score_inputs_prevent": {
        "long_name": "Prevent Risk Factors",
        "short_name": "Prevent",
    },
    "diabetes": {"long_name": "Diabetes", "short_name": "D"},
    "icd_codes": {"long_name": "ICD Codes (ICD)", "short_name": "ICD"},
    "physical_measures": {"long_name": "Physical Measures (PM)", "short_name": "PM"},
    "lifestyle_and_environment": {
        "long_name": "Lifestyle and Environment (LE)",
        "short_name": "LE",
    },
    "family_history": {"long_name": "Family History (FH)", "short_name": "FH"},
    "sociodemographics": {
        "long_name": "Sociodemographics (SD)",
        "short_name": "SD",
    },
    "polygenic_risk_scores_subset": {
        "long_name": "Polygenic Risk Scores Subset",
        "short_name": "PRS_sub",
    },
    "polygenic_risk_scores_all": {
        "long_name": "Polygenic Risk Scores (PRS)",
        "short_name": "PRS",
    },
    "medical_history_all": {"long_name": "Medical History (MH)", "short_name": "MH"},
    "smoking": {"long_name": "Smoking", "short_name": "Sm"},
    "sleep": {"long_name": "Sleep", "short_name": "Sl"},
    "physical_activity": {"long_name": "Physical Activity (PA)", "short_name": "PA"},
    "alcohol": {"long_name": "Alcohol (A)", "short_name": "A"},
    "blood_samples": {"long_name": "Blood Samples (BS)", "short_name": "BS"},
    "urine_assays": {"long_name": "Urine Assays (UA)", "short_name": "UA"},
    ##
    "base_risk_score_inputs_transformed": {
        "long_name": "Framingham Risk Factors (Transformed)",
        "short_name": "Fram (T)",
    },
    "base_risk_score_inputs_transformed_bp": {
        "long_name": "Framingham Risk Factors (Transformed)",
        "short_name": "Fram (T)",
    },
    "base_risk_score_inputs_transformed_ch": {
        "long_name": "Framingham Risk Factors (Transformed)",
        "short_name": "Fram (T)",
    },
    "additional_risk_score_inputs_aha_acc_transformed": {
        "long_name": "Ethnic Background (Transformed)",
        "short_name": "EB (T)",
    },
    "additional_risk_score_inputs_prevent_transformed": {
        "long_name": "Prevent Risk Factors (Transformed)",
        "short_name": "Prevent (T)",
    },
}

model_names_short = {
    "Framingham Risk Score": "Framingham",
    "PREVENT Risk Score": "PREVENT",
    "ACC/AHA Risk Score": "ACC/AHA",
    "QRISK Risk Score": "QRISK",
    "SCORE Risk Score": "SCORE",
    "Cox PH Model": "Cox PH",
    "Logistic Regression": "LogReg",
    "Gradient Boosted Trees": "GBTs",
}
