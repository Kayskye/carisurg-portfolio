# Benchmark Comparison Table
**Project:** A Bias-Audited ML Triage Tool — CariSurg MedPath
**Student:** Kaylah Leigertwood-Ollivierre

All models evaluated on the same 20% stratified test set (random_state=42, n=11,025 patients).

**Primary metric:** Macro F1 — equal weight to all ESI classes.

| Model | Accuracy | Macro F1 | Weighted F1 | ESI 1 Recall | ESI 1 F1 | ESI 2 F1 | ESI 3 F1 | ESI 4 F1 | ESI 5 F1 |
|---|---|---|---|---|---|---|---|---|---|
| Stratified Random Baseline | 0.3754 | 0.2039 | 0.3746 | 0.0 | 0.0 | 0.3306 | 0.4944 | 0.1475 | 0.0469 |
| Logistic Regression (Week 6) | 0.262 | 0.2124 | 0.3221 | 0.8125 | 0.0109 | 0.3558 | 0.3248 | 0.2805 | 0.0901 |
| Decision Tree depth=10 (Week 6) | 0.3257 | 0.2307 | 0.3579 | 0.1875 | 0.0107 | 0.5283 | 0.3002 | 0.2301 | 0.0843 |
| Random Forest (Week 7) | 0.4387 | 0.2828 | 0.439 | 0.0 | 0.0 | 0.5862 | 0.3798 | 0.3759 | 0.072 |

### Notes
- ESI 1 Recall: proportion of life-threatening patients correctly identified (the most clinically critical single measure).
- Macro F1: unweighted average across all five ESI classes — cannot be inflated by majority-class performance.
- Weighted F1: average weighted by class frequency — higher than macro F1 when majority-class performance is strong.
- All baselines use class_weight='balanced' to correct for ESI class imbalance.
