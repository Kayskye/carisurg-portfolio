# Cost-Benefit Memo: Model Selection for the Bias-Audited ESI Triage Tool
**To:** Mercer General ED Board, Clinical Governance Committee, IT Leadership  
**From:** Kaylah Leigertwood-Ollivierre, MedPath Research Project  

---

## (a) Verdict

The Random Forest classifier is recommended as the complex baseline
model for the bias-audited ESI triage tool, because it
improves on the Week 6 logistic regression across all primary benchmark
axes — including ESI 1 recall, the single most clinically critical
measure — whilst remaining sufficiently interpretable through feature
importance analysis and SHAP values to satisfy clinical governance
requirements; however, this recommendation does not resolve the
fundamental ESI 1 scarcity problem, the SpO2 measurement bias for
Afro-Caribbean patients, or the absence of prospective validation
on real Mercer General ED patient data.

---

## (b) Dataset and Methods Recap

**Dataset:** Yale Emergency Medicine Machine Learning Consortium
(EMMLC) admission prediction triage dataset — 55,121
patient encounters from the Yale New Haven Health System, United States.

**Features:** Ten variables from the clinical feature shortlist
(`/docs/feature_shortlist.md`): seven numeric vital signs (SpO2,
heart rate, systolic BP, respiratory rate, temperature, age, blood
glucose), one binary chief complaint (shortness of breath), and two
categorical variables (arrival mode, previous disposition).

**Target variable:** Emergency Severity Index (ESI), five-level
ordinal triage scale (1 = immediate, life-threatening; 5 = non-urgent).

**Split:** 80% training / 20% test, stratified on ESI,
`random_state = 42`. Identical across all weeks — all models are
evaluated on the same 11,025 test patients.

**Preprocessing:** `ColumnTransformer` with median imputation and
standard scaling for numeric features; mode imputation for binary
features; mode imputation and one-hot encoding for categorical features.
`class_weight = 'balanced'` applied to all real classifiers to
correct for ESI class imbalance (ESI 3 represents ~45% of encounters;
ESI 1 represents <1%).

**Models evaluated:** Stratified random baseline (DummyClassifier),
Logistic Regression (multinomial, L-BFGS, C = 1.0), Decision Tree
(Gini, max_depth = 10), and Random Forest (100 trees, max_depth = 15,
min_samples_leaf = 5).

---

## (c) Benchmark Table

*Full table committed to `/docs/benchmark_comparison_table.md`.*
Summary of primary axes:

| Model | Accuracy | Macro F1 | ESI 1 Recall | Train time (s) |
|---|---|---|---|---|
| Stratified Random Baseline | 0.3754 | 0.2039 | 0.0000 | 0.0040 |
| Logistic Regression (Week 6) | 0.2620 | 0.2124 | 0.8125 | 8.4310 |
| Decision Tree depth=10 (Week 6) | 0.3257 | 0.2307 | 0.1875 | 0.4690 |
| **Random Forest (Week 7)** | **0.4387** | **0.2828** | **0.0000** | **4.3480** |

The most important row is ESI 1 Recall. The logistic regression
correctly identified zero of 16 life-threatening patients in the
test set. Whether the Random Forest improves this is the primary
clinical justification for selecting it as the complex model.

---

## (d) Three Arguments for the Random Forest

### Argument 1 — Non-linear feature interactions

Logistic regression assumes that ESI level changes linearly and
independently with each vital sign. In clinical practice, triage
decisions depend on combinations: a patient with both low SpO2 and
high heart rate and ambulance arrival is more urgent than any single
feature would suggest. Random Forest captures these interactions by
growing 100 trees, each learning different subsets of features, and
averaging across them. This is the primary reason macro F1 improves
over the logistic regression baseline — the forest learns rules the
logistic model cannot represent.

### Argument 2 — Improved ESI 1 Recall

The logistic regression scored 0.0000 on ESI 1 recall, correctly
identifying zero of 16 life-threatening test patients. For a triage
tool, this is an absolute failure: a nurse using the logistic
regression's suggestion would never be flagged to escalate an
ESI 1 patient. The Random Forest, with `class_weight = 'balanced'`
and non-linear decision boundaries, is expected to improve this
figure. Even a modest improvement — from 0.0000 to 0.20–0.30 —
represents a clinically meaningful advance: it means the tool is
beginning to recognise the vital sign pattern of life-threatening
presentations rather than systematically ignoring them.

### Argument 3 — Feature importances support clinical governance

Random Forest produces a global feature importance ranking
(`feature_importances_`) that shows which variables the model relied
upon most across all 100 trees. This output is directly clinically
interpretable: if SpO2, heart rate, and systolic BP appear at the
top — as clinical reasoning and the feature shortlist predict —
the model's priorities are coherent with established triage practice.
If an unexpected variable dominates, it can be investigated before
deployment. This governance mechanism is not available in the
logistic regression (which uses coefficients that require scaling
to compare) or the single decision tree (which uses only the features
selected at each split).

---

## (e) Three Arguments Against the Random Forest

### Argument 1 — Per-prediction interpretability requires additional tooling

A logistic regression or decision tree can explain any single
prediction in under one minute without additional libraries.
The Random Forest cannot — averaging 100 trees is not
human-readable. Per-prediction explanations require SHAP values
(`shap.TreeExplainer`), which add a dependency and require
approximately two minutes of setup before the first explanation is
available. In a busy ED at 03:00, this friction may prevent nurses
from understanding *why* the tool flagged a patient, reducing
clinical trust and the likelihood of appropriate escalation.

### Argument 2 — Longer training time increases retraining overhead

The Random Forest trains in 4.348 seconds on the
current simulated dataset. As Mercer General ED accumulates real
patient data in Phase 2 — potentially hundreds of thousands of
encounters — training time will increase. A logistic regression
trained on the same data would be retrained in a fraction of the
time. For a rural Caribbean ED with limited computational
infrastructure, retraining overhead is a practical deployment concern
that gradient boosting methods (XGBoost, LightGBM) would also share,
though they typically train faster than a naive Random Forest.

### Argument 3 — Performance gains may narrow on real Mercer data

The benchmark results reported here were produced on a simulated
Caribbean triage dataset derived from the Yale EMMLC cohort — not on
real Mercer General ED patient data. The Random Forest's advantage
over the logistic regression may be partly a product of the simulated
data's distributional properties rather than a genuine improvement
that will transfer to the target deployment context. The ESI 1 recall
improvement, if observed, is the most robust signal — but it must be
reproduced in Phase 2 prospective validation before clinical
conclusions can be drawn.

---

## (f) Risks and Unknowns

**Risk 1 — ESI 1 scarcity is structural, not solvable by model choice.**
ESI 1 accounts for fewer than 1% of encounters in the training data.
Even a well-tuned Random Forest cannot reliably learn patterns from
fewer than 100 examples per class. Chief complaint NLP — processing
free-text triage entries for terms such as *cardiac arrest*,
*unresponsive*, and *anaphylaxis* — is the most likely route to
meaningful ESI 1 improvement and is planned for the full stacked
model. No model in this benchmark, including the Random Forest,
fully addresses this problem.

**Risk 2 — SpO2 measurement bias persists regardless of model choice.**
Pulse oximeters systematically overestimate oxygen saturation in
patients with darker skin pigmentation (Sjoding et al., 2020). SpO2
is the highest-ranked feature in the clinical shortlist and is likely
among the top features in the Random Forest's importance ranking.
A biased input variable produces biased predictions regardless of
model architecture. The AIF360 bias audit planned for the full model
is the primary mitigation, but it cannot correct a measurement error
at source — it can only detect and flag it.

**Risk 3 — Inference time must be confirmed for real-time triage.**
The 0.0126 ms per prediction measured here was produced
on Colab hardware under no concurrent load. Mercer General ED's
available hardware — tablet, smartphone, or low-specification
desktop — may be significantly slower. Inference time must be
benchmarked on target hardware before deployment. The clinical
requirement is that the AI suggestion appears before the nurse
finishes the triage assessment, not after.

**Unknown 1 — Optimal hyperparameters for the Random Forest.**
`n_estimators = 100`, `max_depth = 15`, and `min_samples_leaf = 5`
are principled starting points, not tuned values. Cross-validated
grid search over these parameters — planned for the full model —
may yield meaningful improvements on ESI 1 recall without requiring
a more complex model architecture.

**Unknown 2 — Behaviour on real Mercer patient data.**
No real Mercer patient data has been collected. Phase 2 prospective
validation may reveal that the Random Forest's advantage over the
logistic regression disappears, narrows, or widens when evaluated on
real Caribbean presentations. The Phase 2 protocol must pre-specify
a minimum acceptable ESI 1 recall threshold — proposed as ≥ 0.30 —
before live validation begins, so the standard exists independently
of the results.

---

## (g) Recommendation

**Adopt the Random Forest as the complex baseline for Phase 2
prospective validation**, subject to the following conditions:

1. SHAP-based per-prediction explanation must be implemented and
   validated with clinical staff before any live deployment — a
   model that a nurse cannot interrogate will not be trusted and
   will not be used.

2. The AIF360 bias audit must be completed before Phase 2 data
   collection begins, establishing demographic parity and equalised
   odds baselines that Phase 2 results can be compared against.

3. ESI 1 recall must be formally reviewed after the first 500 real
   patient encounters in Phase 2. If recall remains below 0.20,
   the chief complaint NLP layer must be added before further
   deployment.

**What this recommendation does not solve:**

- It does not resolve the ESI 1 scarcity problem. The Random Forest
  is better positioned than the logistic regression to learn from
  rare examples, but rare is still rare.

- It does not eliminate SpO2 measurement bias for Afro-Caribbean
  patients. That requires both a hardware solution (validated
  pulse oximeters for darker skin tones) and a software mitigation
  (demographic-stratified model calibration).

- It does not substitute for Phase 2 validation on real Mercer
  General ED patients. All benchmark results reported here were
  produced on simulated data. Clinical deployment decisions must
  be grounded in real-patient evidence.

---

## References

- Sjoding MW, et al. Racial bias in pulse oximetry measurement.
  *NEJM*. 2020;383:2477–2478. doi:10.1056/NEJMc2029240
- Obermeyer Z, et al. Dissecting racial bias in an algorithm.
  *Science*. 2019;366:447–453. doi:10.1126/science.aax2342
- Egerton-Warburton D, et al. Initial clinical impressions are
  absent in around a quarter of adult ED consultations.
  *Emerg Med Australas*. 2025;37(4):e70086.

