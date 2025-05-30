# Pooled Cohort Risk Score (2013 ACC/AHA)

Reference: [Goff et al. (2013)](https://pubmed.ncbi.nlm.nih.gov/24222018/)
Developed by the Amercian College of Cardiology and the American Heart Association

## Introduction

- Guidelines for Risk Assessment
- Developed from several long-standing population-based cohort studies
- Risk Estimation is based on group averages (*pooled cohorts*) --> Pooled Cohort Equations
- Modification and adoption of the Framingham 10-year risk score
- Sex- and race-specific estimates for African-American and white men and women aged 40-79

## Algorithm

- Risk Target: 10-year risk of developing a **first** ASCVD event, defined as myocardial infarction or coronary heart disease (CHD) death, fatal or nonfatal stroke, among people free from ASCVD at the beginning of the period.
- Proportial-hazards models
- Equation parameters (coefficients) are provided in Table A in [Goff et al. (2013)](https://pubmed.ncbi.nlm.nih.gov/24222018/)


### Variables

- Sex
- Race
- Age
- Total cholesterol
- High-density lipoprotein cholesterol (HDL cholesterol)
- Systolic blood pressure (including treated or untreated status)
- Diabetes mellitus
- Current Smoking Status

### Recommendations

- Non-Hispanic African Americans and non-Hispanic whites, 40-79: Use race- and sex-specific Pooled Cohort Equations
- Other than African Americans and non-Hispanic whites: only sex-specific Pooled Cohort Equations (However, they did not provide the coefficients (?))

## Baseline

- [Han et al., iScience 27, (2024)](https://www.cell.com/iscience/pdf/S2589-0042(24)00243-8.pdf) use this risk score as a baseline (ACC/AHA). Subsample: Excluded participants with missing data. Excluded participants with previous MACE event. Randomly selected 50k. [Github Repo](https://github.com/CMI-Laboratory/GPTCVD).
