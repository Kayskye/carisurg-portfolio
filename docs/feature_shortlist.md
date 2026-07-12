# Top-10 Feature Shortlist — Triage Level (ESI) Prediction
**Document path:** `/docs/feature_shortlist.md`  
**Project:** A Bias-Audited ML Triage Tool for Nurse-Led Emergency Care in a Caribbean Regional ED  
**Student:** Kaylah Leigertwood-Ollivierre  
**Date:** [Date]  

---

## Selection Methodology

Features are ranked by a combination of two criteria:

1. **Clinical primacy** — whether the variable is a direct determinant of ESI
   assignment under the five-level triage protocol (e.g. danger-zone vitals
   automatically trigger ESI 2)
2. **Exploratory evidence** — Spearman correlation with `esi` from the profiling
   notebook, or documented association with high-acuity presentation in the
   chief complaint analysis

Features were excluded if they:
- Are identifiers or administrative artefacts (e.g. `Unnamed: 0`)
- Encode information unavailable at the triage moment (e.g. final diagnosis)
- Have missingness greater than 40% without a clinical imputation strategy

---

## Feature Shortlist

| Rank | Feature name | Clinical label | Justification |
|---|---|---|---|
| 1 | `triage_vital_o2` | SpO2 on arrival (%) | SpO2 below 92% is a designated danger-zone vital in the ESI protocol that automatically triggers ESI 2 assignment; the profiling notebook confirms a negative Spearman correlation with ESI (lower SpO2 = higher acuity) consistent with this clinical rule. |
| 2 | `triage_vital_hr` | Heart Rate on arrival (bpm) | Heart rate above 100 bpm or below 50 bpm indicates haemodynamic instability and is a primary criterion for ESI 2 assignment; the violin plot (Figure 6) shows a clear separation in heart rate distributions across ESI levels. |
| 3 | `triage_vital_sbp` | Systolic Blood Pressure (mmHg) | Systolic BP below 90 mmHg indicates shock — a life-threatening condition requiring ESI 1 or 2 response; the correlation heatmap (Figure 5) shows systolic BP among the vital signs most strongly correlated with triage level. |
| 4 | `triage_vital_rr` | Respiratory Rate (bpm) | Respiratory rate above 20 bpm signals respiratory compromise and is a danger-zone vital for ESI 2; the profiling notebook identifies it as one of the most negatively correlated vitals with ESI level, consistent with published triage literature. |
| 5 | `triage_vital_temp` | Temperature on arrival (°C) | Fever above 38.5 °C or hypothermia below 36 °C indicate systemic illness requiring multiple resources; distinct ESI-stratified temperature distributions are visible in the boxplot analysis (Figure 2). |
| 6 | `age` | Patient age (years) | Older patients require more clinical resources per presentation, are more likely to deteriorate, and have higher admission rates; age correlates negatively with ESI level in the dataset, and dedicated triage categories exist for specific age thresholds (fall >65 years, fever ≥75 years) in the chief complaint columns. |
| 7 | `arrivalmode` | Arrival mode (ambulance / walk-in / other) | Ambulance arrivals are systematically associated with ESI 1–3 presentations because patients who are transported by EMS have already been assessed as requiring urgent care; the disposition distribution analysis (Figure 4) shows arrival mode is one of the strongest demographic predictors of admission. |
| 8 | `triage_glucose` | Blood glucose on arrival (mmol/L) | Hypoglycaemia (below 3.9 mmol/L) and hyperglycaemia (above 11.1 mmol/L) are metabolic emergencies requiring immediate intervention; despite high missingness, glucose is clinically indispensable for diabetic presentations, which constitute a significant proportion of Caribbean ED attendances. |
| 9 | `cc_shortnessofbreath` | Chief complaint: shortness of breath (binary) | Respiratory chief complaints are among the strongest predictors of high-acuity triage assignment; the chief complaint analysis confirms shortness of breath appears in a disproportionate fraction of ESI 1–2 encounters, and the clinical mechanism is direct — dyspnoea triggers airway and breathing assessment which governs ESI scoring. |
| 10 | `previousdispo` | Previous ED disposition (categorical) | A patient who was admitted on their previous ED visit is significantly more likely to present with a high-acuity condition on return; this variable encodes prior clinical complexity in a structured form and shows meaningful correlation with ESI level in the exploratory analysis. |

---

## Notes on Excluded Features

- **`triage_vital_dbp`** (Diastolic BP) — carries clinical information but is
  highly correlated with systolic BP (Figure 5 heatmap); including both risks
  multicollinearity without proportionate predictive gain.

- **`triage_vital_o2_device`** — clinically meaningful (patients on oxygen
  are more acutely unwell) but missingness is high because the field is only
  recorded when supplemental oxygen is in use; requires careful imputation
  before inclusion in a final model.

- **Chief complaint binary columns (general)** — the 180+ binary columns
  collectively have predictive value, but individually many are sparse; a
  dimensionality reduction step (e.g. selecting complaints with >5% prevalence
  and >30% ESI 1–2 association rate) should precede inclusion.

---

## Equity Consideration

SpO2 (`triage_vital_o2`) is ranked first on clinical grounds but carries a
documented measurement bias against patients with darker skin pigmentation
(Sjoding et al., 2020). Pulse oximeters systematically overestimate oxygen
saturation in Afro-Caribbean patients, which would cause the model to
underestimate hypoxaemia severity in this group. The bias audit (AIF360)
must explicitly test whether SpO2-driven ESI predictions are equitable
across demographic subgroups, and a borderline SpO2 alert (94–96%) must
be implemented in the nurse-facing interface regardless of the ML output.

---

## References

- Sjoding MW, et al. Racial bias in pulse oximetry measurement. *NEJM*. 2020;383:2477–2478. doi:10.1056/NEJMc2029240
- Obermeyer Z, et al. Dissecting racial bias in an algorithm. *Science*. 2019;366:447–453. doi:10.1126/science.aax2342
- Araouchi Z, Adda M. TriageIntelli. *Procedia Comput Sci*. 2024;251:430–437. doi:10.1016/j.procs.2024.11.130
