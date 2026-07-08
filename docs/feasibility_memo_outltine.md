# Feasibility Memo Outline

---

## (a) One-Sentence Verdict

The Yale Emergency Department triage dataset is suitable for developing a machine learning-based admission prediction model because it contains clinically rich demographic, triage, and chief complaint data across 55,121 patient encounters — however, moderate missingness in vital sign variables, a substantial outlier burden in laboratory values, and its origin from a single North American health system require active data quality measures and a Phase 2 local validation study before findings can be applied at Mercer General ED.

---

## (b) Dataset Summary

**What the dataset contains:**

- **55,121** emergency department encounters collected across multiple sites within the Yale New Haven Health System, United States
- **226** total columns organised into five clinical categories:
  - *Triage acuity* — Emergency Severity Index (ESI 1–5), the same instrument used at Mercer General ED
  - *Vital signs* — heart rate, systolic/diastolic BP, respiratory rate, SpO2, temperature, blood glucose
  - *Demographics* — age, sex, race, ethnicity, language, marital status, employment, insurance
  - *Arrival data* — arrival mode, month, day, hour band, previous disposition
  - *Chief complaints* — **200** binary indicator columns, one per presenting complaint
- **Primary outcome variable:** `disposition` — the patient's final ED outcome (admitted, discharged, transferred, left without being seen)
- **ESI distribution:** ESI 1: 0.1%, ESI 2: 32.5%, ESI 3: 49.0%, ESI 4: 16.1%, ESI 5: 2.2%
- **Disposition split:** Discharged: 62.7%, Admitted: 37.3%

**Why this is appropriate for admission prediction:**  
The dataset captures the exact triage variables collected on the Mercer General ED paper intake form — ESI level, vital signs, and chief complaint — making it structurally compatible with the proposed ML triage tool without requiring additional data collection at the point of care.

> *See Figure 3 (ESI distribution) and Figure 4 (Disposition outcomes) from the profiling notebook.*

---

## (c) Top 3 Quality Concerns

###Concern 1 — Absence of Missing Values Suggests Pre-Release Record Exclusion
The profiling analysis found zero missing values across all 226 columns — an outcome that is implausible in a real clinical dataset, where incomplete documentation is endemic. Egerton-Warburton et al. (2025) report absent clinical documentation in up to 67% of low-acuity ED encounters; a dataset with no missingness at all has almost certainly been pre-processed to exclude incomplete records before public release.
This matters because excluded records are not random. Patients who leave without being seen, arrive in extremis with abbreviated documentation, or present during surge periods when recording is rushed are disproportionately likely to have incomplete records — and disproportionately likely to be high-acuity or from marginalised demographic groups. Removing these records before training introduces a selection bias that is invisible in the cleaned dataset but would affect model performance on exactly the patients where accurate triage prediction is most consequential.
**Design response:** The provenance of the public dataset release should be documented in the project methods section. Phase 2 validation on real Mercer patient data — which will include genuine missingness — is essential to test whether model performance degrades when encountering incomplete records.
---

### Concern 2 — Outliers in Vital Sign Variables

**Observation:** IQR-based outlier detection identified a significant number of statistically extreme values across vital sign columns, particularly in heart rate, blood glucose, and temperature (see Figure 2 — boxplots from profiling notebook).

**Clinical interpretation:** Two distinct categories must be distinguished:
- *Clinically valid extremes* — e.g. HR 180 bpm in supraventricular tachycardia; glucose 40 mmol/L in diabetic ketoacidosis. These must be **retained** — they represent the high-acuity presentations where accurate triage prediction is most consequential.
- *Recording errors* — e.g. HR = 0, SpO2 > 100%, temperature = 0 °C. These must be **removed**, but ideally after clinical review rather than automated deletion.

**Implication for this project:** Automated removal using an IQR threshold would silently discard genuinely sick patients. A clinical review step is required before finalising the training set.

---

### Concern 3 — Single North American Health System — Demographic and Epidemiological Mismatch

**Observation:** The dataset was collected entirely within the Yale New Haven Health System in Connecticut, USA. The racial composition is dominated by White, Black/African American, and Hispanic patients (see Figure 5 from profiling notebook). Indo-Caribbean presentations — which represent approximately 35–40% of the population in Trinidad and Tobago — are absent entirely.

**Clinical interpretation:** Disease prevalence, presentation patterns, and care-seeking behaviour differ systematically between a Yale urban academic ED and a Caribbean regional ED. Conditions with higher prevalence in Caribbean populations (sickle cell crisis, dengue fever, type 2 diabetes complications, specific cardiovascular profiles) are underrepresented in the training data.

**Implication for this project:** A model trained exclusively on this dataset and deployed at Mercer General without local validation represents the training-distribution mismatch risk documented in the project risk register (Risk 5). Phase 2 prospective validation using real Mercer patient encounters is required, with a pre-defined performance degradation threshold of ≤10% before live deployment.

---

## (d) Top 3 Reasons to Proceed

### Reason 1 — Triage Variables Directly Aligned with the Mercer Intake Form

The dataset contains precisely the variables captured on Mercer's existing paper triage form: ESI level, vital signs (HR, BP, RR, SpO2, temperature, glucose), arrival mode, and chief complaint. A model trained on this data can be architecturally designed to accept Mercer triage form inputs without requiring any additional data collection at the point of care. For a triage nurse operating within a 3–5 minute assessment window, a tool that works with existing inputs is the only tool that will be adopted in practice. This structural alignment is a decisive advantage over datasets from settings with different triage instruments.

---

### Reason 2 — Large, Diverse Sample Enabling Robust Training and Bias Auditing

The dataset contains 55,121 encounters — large enough to support training, validation, and test splits while retaining statistical power across demographic subgroups. This sample size allows the AIF360 bias audit to be run with meaningful subgroup sample sizes, producing equalised odds and demographic parity metrics that are statistically reliable rather than noise-dominated. Published AI triage models trained on comparable datasets have reported weighted F1 scores above 80% across ESI classes (Araouchi & Adda, 2024); the stacking ensemble architecture proposed for this project has a credible evidence base at this scale.

---

### Reason 3 — Clinically Meaningful Outcome Variable Directly Relevant to ED Overcrowding

The `disposition` variable captures the real-world decision that drives boarding and overcrowding — whether a patient requires admission. Accurate admission prediction at triage allows the charge nurse to anticipate bed demand, flag patients likely to need admission before a bed becomes unavailable, and reduce the boarding delay that the Mercer operational brief identifies as a primary source of overcrowding. Tyler et al. (2024), reviewing 29 primary studies, confirmed that ML models trained to predict ED disposition consistently outperform conventional protocols in forecasting admission rates and length of stay.

---

## (e) Caveats

The following limitations must be disclosed to the Board before any deployment decision:

- **Single health system:** All data from Yale New Haven Health System. Performance metrics may not replicate in a Caribbean regional ED with different patient populations, disease prevalence, or triage protocols.

- **Temporal drift:** Dataset collected over a defined historical period. Seasonal presentation patterns, available treatments, and triage practices change over time. Retraining must be triggered when accuracy declines more than 10% from the validation benchmark.

- **Demographic representation gap:** Indo-Caribbean patients absent from the dataset. The AIF360 bias audit cannot evaluate subgroups that are not represented. Phase 2 local data collection must prioritise capturing Indo-Caribbean presentations.

- **Missing laboratory data:** Glucose and SpO2 device data absent for a substantial proportion of encounters. Predictions from incomplete vital records must be flagged to the triage nurse rather than presented at the same confidence as complete-data predictions.

- **Observational data from routine care:** Dataset not collected for research purposes. Yale-specific documentation conventions, coding practices, and triage workflows may not transfer directly to Mercer General ED.

- **Chief complaint sparsity:** 200 binary chief complaint columns, most of which are zero for the majority of encounters. High-dimensional sparse features increase model complexity and the risk of overfitting on rare presentations.

---

## Supporting Figures

| Figure | File | Key observation |
|---|---|---|
| Figure 1a | `fig1a_missingness_matrix.png` | Pattern of missing data across encounters |
| Figure 1b | `fig1b_missingness_barchart.png` | Columns ranked by missingness percentage |
| Figure 2 | `fig2_vitals_boxplots.png` | Outlier burden in vital sign variables |
| Figure 3 | `fig3_esi_distribution.png` | Acuity mix of the training cohort |
| Figure 4 | `fig4_disposition.png` | Outcome balance — admitted vs discharged |
| Figure 5 | `fig5_race_distribution.png` | Racial composition — equity context |

*All figures generated by `/notebooks/Week5_Data_Profiling.py` and committed to `/docs`.*

---

## References

1. Egerton-Warburton D, et al. *Emerg Med Australas*. 2025;37(4):e70086. doi:10.1111/1742-6723.70086
2. Teeple S, et al. *JAMIA Open*. 2023;6(4):ooad107. doi:10.1093/jamiaopen/ooad107
3. Huang J, et al. *JMIR Med Inform*. 2022;10(5):e36388. doi:10.2196/36388
4. Araouchi Z, Adda M. *Procedia Comput Sci*. 2024;251:430–437. doi:10.1016/j.procs.2024.11.130
5. Tyler S, et al. *Cureus*. 2024;16(5). doi:10.7759/cureus.59906
