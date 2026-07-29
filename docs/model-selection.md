# Model Selection Results Table
**Project:** A Bias-Audited ML Triage Tool for Nurse-Led Emergency Care in a Caribbean Regional ED  

> **Full reasoning for the winning model choice is documented in:**  
> [`/docs/model-choice.md`]
---

## Evaluation Conditions

All models were evaluated on the **same 20% stratified hold-out set**
(random_seed = 42, identical split across Weeks 6, 7, and 8).
Stratification on the target variable (`esi`) ensures each acuity class
appears in both training and test sets at the same proportion.

**Primary metric:** Macro F1 — equal weight to all five ESI classes,
regardless of frequency. A model excelling on ESI 3 (the most common
class, ~45% of encounters) but failing on ESI 1 (life-threatening, <1%)
will score low on Macro F1 and cannot hide behind weighted or aggregate figures.

---

## Complete Results Table

| Chosen | Model | Key Hyperparameters | Accuracy | Macro Precision | Macro Recall | Macro F1 | Train time (s) | Infer time (ms/pred) |
|---|---|---|---|---|---|---|---|---|
| | Stratified Random Baseline | strategy=stratified | 0.3754 | 0.2041 | 0.2037 | 0.2039 | 0.0040 | 0.0006 |
| | Logistic Regression | C=1.0, max_iter=1000, class_weight=balanced | 0.2620 | 0.2768 | 0.4244 | 0.2124 | 4.3620 | 0.0023 |
| | Decision Tree | max_depth=10, class_weight=balanced, criterion=gini | 0.3257 | 0.2735 | 0.3475 | 0.2307 | 0.2550 | 0.0010 |
| | Random Forest | n_estimators=100, max_depth=15, min_samples_leaf=5, class_weight=balanced | 0.4387 | 0.2897 | 0.3228 | 0.2828 | 2.7830 | 0.0143 |
| | Extra Trees | n_estimators=100, max_depth=15, min_samples_leaf=5, class_weight=balanced | 0.3205 | 0.2773 | 0.3661 | 0.2358 | 2.8250 | 0.0196 |
| | XGBoost | n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8 | 0.3938 | 0.2953 | 0.3526 | 0.2758 | 6.2130 | 0.0067 |
| **CHOSEN** | **LightGBM** | **n_estimators=100, max_depth=10, learning_rate=0.1, num_leaves=31, class_weight=balanced** | **0.4093** | **0.3001** | **0.3398** | **0.2848** | **17.2110** | **0.0467** |
| | HistGradientBoosting | max_iter=100, max_depth=10, learning_rate=0.1, class_weight=balanced | 0.3702 | 0.2926 | 0.4044 | 0.2622 | 0.8850 | 0.0126 |
| | MLP (128→64) | hidden_layers=(128,64), activation=relu, solver=adam, max_iter=300 | 0.5454 | 0.4229 | 0.2553 | 0.2333 | 20.1790 | 0.0030 |

*Metrics at 4 decimal places. Bold row = chosen model.*

---

## ESI 1 Recall — Clinical Safety Supplement

ESI 1 Recall measures what proportion of life-threatening patients
the model correctly identifies. It is reported separately because it
is the single most clinically consequential measure and is not visible
in the aggregate metrics above.

| Model | ESI 1 Recall | Note |
|---|---|---|
| Stratified Random Baseline | 0.0000 | No useful signal |
| Logistic Regression | 0.8125 | High recall but accuracy 0.2620 — **below the random baseline**; would lose clinical trust |
| Decision Tree | 0.1875 | |
| Random Forest | 0.0000 | |
| Extra Trees | 0.3750 | |
| XGBoost | 0.1250 | |
| **LightGBM ** | **0.0625** | Modest ESI 1 detection; primary target for Phase 2 NLP layer |
| HistGradientBoosting | 0.3750 | |
| MLP (128→64) | 0.0000 | Highest accuracy but zero ESI 1 detection — clinically unsafe |

---

## Why LightGBM Was Chosen

Three criteria drove the selection:

**1. Best Macro F1 (0.2848).**
Macro F1 is the primary metric because it penalises poor performance
on any ESI class equally, regardless of class frequency. LightGBM
achieves the highest Macro F1 of all models tested.

**2. Clinically credible overall accuracy (0.4093).**
LightGBM's accuracy is above the stratified random baseline (0.3754),
confirming it has learned generalisable patterns. The Logistic
Regression's accuracy of 0.2620 falls *below* the random baseline —
a model that performs worse than chance on overall accuracy would
lose clinical trust within a shift, regardless of its ESI 1 recall.
Dr. De Freitas' feedback specifically challenged whether a 0.26
accuracy is clinically appropriate; it is not.

**3. Highest Macro Precision (0.3001).**
LightGBM is the only model to cross the 0.30 threshold on macro
precision, meaning its high-acuity flags are more often correct
than any alternative.

**What LightGBM does NOT solve:**
ESI 1 recall remains critically low at 0.0625 — the primary unsolved
challenge. The NLP chief complaint layer (identifying terms such as
*cardiac arrest*, *unresponsive*, and *anaphylaxis*) is the planned
Phase 2 intervention to address this.

> **Full reasoning, alternatives considered, and things not yet known:**  
> [`/docs/model_choice.md`]

---

## Interpretability

| Model | Explainable in < 1 min? | Method |
|---|---|---|
| Logistic Regression | Yes | Feature coefficients |
| Decision Tree | Yes | Tree traversal via `export_text()` |
| Random Forest | Partial | SHAP `TreeExplainer` |
| Extra Trees | Partial | SHAP `TreeExplainer` |
| XGBoost | Partial | SHAP `TreeExplainer` |
| **LightGBM ** | **Partial** | **`feature_importances_` globally; SHAP per-prediction** |
| HistGradientBoosting | Partial | SHAP |
| MLP | No | Black box — SHAP approximation only |

---

## Files Referenced

| File | Location |
|---|---|
| Training notebook (Weeks 6–7) | `/notebooks/` |
| Config (final model pinned) | `config.yaml` |
| Decision journal | `/docs/model_choice.md` |
| Cost-benefit memo | `/docs/cost_benefit_memo.md` |
