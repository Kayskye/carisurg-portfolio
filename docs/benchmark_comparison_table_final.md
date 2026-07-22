# Benchmark Comparison Table
**Project:** A Bias-Audited ML Triage Tool — CariSurg MedPath
**Student:** Kaylah Leigertwood-Ollivierre

All models evaluated on the same 20% stratified test set (random_state=42, n=11,025 patients).

**Primary metric:** Macro F1 — equal weight to all ESI classes.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Train time (s) | Infer time (ms/pred) | ESI 1 Recall | Interpretability |
|---|---|---|---|---|---|---|---|---|
| Stratified Random Baseline | 0.3754 | 0.2041 | 0.2037 | 0.2039 | 0.002 | 0.0002 | 0.0 | N/A — random predictions carry no explanation |
| Logistic Regression (Week 6) | 0.262 | 0.2768 | 0.4244 | 0.2124 | 2.189 | 0.0024 | 0.8125 | Yes (<1 min) — feature coefficients explain each prediction |
| Decision Tree depth=10 (Week 6) | 0.3257 | 0.2735 | 0.3475 | 0.2307 | 0.255 | 0.0011 | 0.1875 | Yes (<1 min) — tree traversal via export_text() |
| Random Forest (Week 7) | 0.4387 | 0.2897 | 0.3228 | 0.2828 | 2.588 | 0.0091 | 0.0 | Global: feature_importances_. Per-prediction: SHAP (<1 min after setup) |

### Notes
- ESI 1 Recall: proportion of life-threatening patients correctly identified (the most clinically critical single measure).
- Macro F1: unweighted average across all five ESI classes — cannot be inflated by majority-class performance.
- Weighted F1: average weighted by class frequency — higher than macro F1 when majority-class performance is strong.
- All baselines use class_weight='balanced' to correct for ESI class imbalance.
