# Decision Journal — Week 7 Model Choice
**Decision type:** Algorithm selection for complex model step

---

## Context

- The Week 6 baseline logistic regression correctly identified zero
  of 16 ESI 1 (life-threatening) patients in the test set, scoring
  an ESI 1 recall of 0.0000 on the same 20% stratified hold-out used
  for all subsequent evaluation — a failure mode clinically equivalent
  to a triage tool that never sounds the alarm for cardiac arrest.
- The rubric for Week 7 requires a more complex classifier (random
  forest, gradient boosting, or MLP) trained on the same 80/20
  stratified split (random_state = 42) and evaluated on the same six
  quantitative axes plus one interpretability axis.

---

## Alternatives Considered

- **Gradient Boosting (XGBoost or LightGBM):** Sequential tree
  boosting typically outperforms random forest on tabular data with
  class imbalance; however, it introduces additional hyperparameters
  (learning rate, subsample ratio, number of rounds), is slower to
  tune, and is harder to explain per-prediction without SHAP —
  adding implementation complexity without a clear justification
  at the baseline complex model stage.
- **Small MLP (Multi-Layer Perceptron):** A neural network with
  two hidden layers (e.g. 64 → 32 → 5 output) could capture
  non-linear interactions but requires feature scaling already
  applied in the pipeline, a separate framework decision
  (Keras vs PyTorch), and is the least interpretable option —
  ruling it out on clinical governance grounds given the project's
  bias-audit requirement.
- **Random Forest:** Ensemble of 100 decision trees; scikit-learn
  pipeline-compatible; produces feature importances for global
  interpretability; per-prediction SHAP explanation achievable
  within one minute after initial setup; `class_weight='balanced'`
  directly available; no additional framework dependency beyond
  the existing sklearn stack.

---

## Decision

Random Forest (`sklearn.ensemble.RandomForestClassifier`,
n_estimators = 100, max_depth = 15, min_samples_leaf = 5,
class_weight = 'balanced', random_state = 42) is selected as
the Week 7 complex model.

---

## Reasoning

- **ESI 1 recall is the primary clinical benchmark axis.** The
  logistic regression's 0.0000 ESI 1 recall is the most consequential
  failure in the Week 6 results; Random Forest's non-linear,
  ensemble-averaged decision boundary is more likely to learn the
  rare vital sign patterns associated with ESI 1 presentations than
  either a linear model or a single decision tree (DT macro F1: 0.2307 vs RF macro F1: 0.2828.
- **Interpretability axis is partially satisfied without additional
  libraries.** The global feature importance output from Random Forest
  is immediately available and directly comparable to the ranked
  clinical feature shortlist in `/docs/feature_shortlist.md` — if
  SpO2 and heart rate dominate the importance ranking, the model's
  priorities are clinically coherent; this validation step is not
  possible with MLP or with gradient boosting without SHAP.
- **Same scikit-learn framework as baselines.** Random Forest fits
  directly into the existing `Pipeline` and `ColumnTransformer`
  stack from Week 6 — no new framework, no refactoring, identical
  preprocessing guaranteeing like-for-like comparison on all six
  quantitative axes.

---

## Things Not Yet Known

- Whether the Random Forest's improvement in ESI 1 recall and macro
  F1 over the logistic regression holds when the model is retrained
  on real Mercer General ED patient data in Phase 2, rather than on
  the simulated Yale-derived dataset used here.
- Whether `n_estimators = 100` and `max_depth = 15` are optimal for
  this problem — cross-validated grid search over
  `n_estimators ∈ {50, 100, 200}` and `max_depth ∈ {10, 15, 20}`
  is planned for the full stacked model but was not conducted at the
  baseline complex model stage.
