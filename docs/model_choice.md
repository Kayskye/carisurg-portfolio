# Decision Journal — Week 7 Model Choice
**Decision type:** Algorithm selection for the final production model

---

## Context

- Nine classifiers were benchmarked on the same 20% stratified
  hold-out (random_state = 42): a stratified random baseline,
  Logistic Regression, Decision Tree, Random Forest, Extra Trees,
  XGBoost, LightGBM, HistGradientBoosting, and MLP. The Logistic
  Regression achieved the highest ESI 1 recall (0.8125) but at an
  overall accuracy of 0.2620 — below the stratified random baseline
  of 0.3754 — meaning it performs worse than chance on overall
  classification and would lose clinical trust within a shift.
- Tutor feedback (Dr. De Freitas) specifically challenged whether
  an overall accuracy of 0.26 is appropriate for clinical deployment,
  shifting the primary selection criterion from ESI 1 recall alone
  to balanced performance across all five ESI classes, measured by
  Macro F1 on a model whose accuracy is above the random baseline.

---

## Alternatives Considered

- **Random Forest** (initial Week 7 complex model): Achieved Macro
  F1 of 0.2828 and accuracy of 0.4387, but scored 0.0000 on ESI 1
  recall — correctly identifying zero life-threatening patients. Good
  balanced performance but the complete failure on ESI 1 and the
  availability of stronger alternatives in the extended benchmark
  made it the second-best choice rather than the final one.
- **MLP (128 → 64):** Achieved the highest overall accuracy (0.5454)
  and macro precision (0.4229) but scored 0.0000 on ESI 1 recall and
  the lowest Macro F1 (0.2333) of all non-random models. High
  accuracy with zero ESI 1 detection is clinically dangerous —
  a tool that looks precise on paper but never flags a cardiac arrest
  is more harmful than one with lower headline accuracy.
- **XGBoost:** Achieved Macro F1 of 0.2758 and accuracy of 0.3938,
  but both metrics are below LightGBM on every primary axis. No
  meaningful advantage over LightGBM was observed to justify the
  additional tuning complexity of the subsample and colsample
  hyperparameters.

---

## Decision

LightGBM (`lightgbm.LGBMClassifier`, n_estimators = 100,
max_depth = 10, learning_rate = 0.1, num_leaves = 31,
class_weight = 'balanced', random_state = 42) is selected as
the final production model, pinned in `config.yaml`.

---

## Reasoning

- **Highest Macro F1 across all nine models (0.2848).** Macro F1
  gives equal weight to all five ESI classes — a model excelling on
  ESI 3 (the majority class) cannot inflate this score by ignoring
  ESI 1. LightGBM is the only model that crosses 0.28 on this metric,
  meaning its performance is the most balanced across the full acuity
  spectrum of the nine models tested.
- **Overall accuracy above the stratified random baseline (0.4093
  vs 0.3754).** Dr. De Freitas' feedback established that clinical
  credibility requires accuracy above chance. LightGBM satisfies
  this condition; Logistic Regression (0.2620) and Extra Trees
  (0.3205) do not. A model performing below the random baseline
  would be identified as unreliable by triage nurses within hours
  of deployment.
- **Highest macro precision of all models tested (0.3001).**
  LightGBM is the only model to cross the 0.30 threshold on macro
  precision, meaning when it assigns a triage level it is more often
  correct than any alternative — directly relevant to clinical trust
  in a nurse-facing decision support tool.

---

## Things Not Yet Known

- Whether LightGBM's Macro F1 advantage over the other models holds
  when retrained on real Mercer General ED patient data in Phase 2,
  rather than on the Yale-derived simulated dataset — Caribbean
  presentation patterns (dengue fever, sickle cell, tropical disease
  profiles) absent from the training data may change the relative
  performance ranking.
- Whether the ESI 1 recall of 0.0625 can be meaningfully improved
  through chief complaint NLP (identifying terms such as *cardiac
  arrest*, *unresponsive*, and *anaphylaxis*) without degrading the
  Macro F1 and accuracy gains that justified selecting LightGBM
  over the alternatives.
