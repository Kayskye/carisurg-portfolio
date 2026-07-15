# Final Baseline Model Report
**Project:** A Bias-Audited ML Triage Tool for Nurse-Led Emergency Care in a Caribbean Regional ED  
**Student:** Kaylah Leigertwood-Ollivierre | MedPath Programme  

---

## (a) Results

### Models Evaluated

Three models were trained and evaluated on an 80/20 stratified split
of the Yale EMMLC emergency department triage dataset (random seed: 42).
Stratification on the target variable (Emergency Severity Index, ESI)
ensures each acuity class is represented proportionally in both training
and test sets — critical because ESI 1 (life-threatening) accounts for
fewer than 1% of presentations.

**Stratified Random Baseline (DummyClassifier)** assigns predictions
by sampling randomly from the training class distribution. It ignores
all clinical features entirely. Any real model must exceed this floor
to demonstrate that it has learned something from patient vital signs.

**Logistic Regression** (multinomial, L-BFGS solver, `class_weight=
'balanced'`, `C=1.0`, `max_iter=1000`) was the interim Week 6 model,
which scored full marks at assessment. Class weighting corrects for
ESI imbalance by penalising errors on rare classes proportionally more.

**Decision Tree** (Gini criterion, `max_depth=10`, `class_weight=
'balanced'`) uses a bounded depth to prevent overfitting. The default
`max_depth=None` grows the tree until all leaves are pure, which
memorises training noise rather than learning generalisable clinical
patterns. A depth of 10 allows the model to capture meaningful
multi-feature interactions — for example, the combination of high
heart rate, low SpO2, and ambulance arrival that characterises
high-acuity presentations — whilst limiting the maximum number of
leaf nodes to 1,024, constraining the tree to rules that can
generalise to unseen patients. Depth 10 is a starting point for
the baseline; cross-validated tuning over depths of 5, 10, 15, and
20 is planned for the full stacked model.

### Performance Summary

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Stratified Random Baseline | 0.3754 | 0.2039 | 0.3746 |
| Logistic Regression | 0.2620 | 0.2124 | 0.3221 |
| Decision Tree (max_depth=10) | 0.3257 | 0.2307 | 0.3579 |


Both real models exceed the stratified random baseline on all three
metrics, confirming that the classifiers have extracted genuine signal
from the triage vital signs, arrival mode, and chief complaint features.
However, the overall accuracy figures are misleading on an imbalanced
dataset — the per-class breakdown reveals a more clinically important
pattern addressed in Section (c).

---

## (b) Metric Justification

### Why accuracy is insufficient

The ESI distribution is heavily skewed towards ESI 2 and ESI 3, which
together account for approximately 75% of encounters. A model that
predicts ESI 3 for every patient would achieve roughly 45% accuracy
whilst completely failing on every other acuity level. Accuracy rewards
majority-class performance and conceals minority-class failure — exactly
the failure mode that matters most in emergency triage.

### Why weighted F1 is also insufficient

Weighted F1 averages each class's F1 score in proportion to the number
of patients in that class. ESI 3 receives approximately 45 times the
weight of ESI 1 in this calculation. A model that classifies every
ESI 3 patient correctly but misses every ESI 1 patient can still report
a high weighted F1. For a triage tool, that is a model which performs
well on non-urgent patients and catastrophically on critical ones —
the precise pattern observed in the Week 6 baseline (see Section c).

### Why macro F1 is the primary metric

Macro F1 averages each class's F1 score with equal weight, regardless
of class size. ESI 1 contributes identically to ESI 3 in the final
score. If the model achieves F1 = 0.00 on ESI 1 — because it correctly
identifies zero life-threatening patients — that zero drags the macro
average down sharply, making the failure impossible to hide behind
good performance on common classes.

The difference between macro F1 and weighted F1 on this dataset is
clinically meaningful: weighted F1 is 0.1097 points higher than macro
F1 for the logistic regression, because ESI 3 (the most common class,
with the highest per-class F1) receives disproportionate weight. This
gap directly quantifies how much the weighted metric flatters a model
that is failing on the rarest and most critical patients.

**Primary metric: Macro F1.** It is the only aggregate metric that
holds the model accountable for its performance on ESI 1 patients
rather than allowing their misclassification to be averaged away.

---

## (c) Failure-Mode Reflection

### Which class the model misses most

The logistic regression correctly identified **zero of 16 ESI 1
patients** in the test set. All 16 were misclassified — 11 as ESI 2
and 5 as ESI 3. The model has learned no reliable pattern for the
most time-critical acuity class.

The decision tree's confusion matrix should be examined for the same
pattern. Even if it improves on ESI 1 recall, the failure mode is
sufficiently severe that it requires explicit disclosure in any
clinical deployment decision.

### What missing ESI 1 means for a real patient

Consider a 58-year-old man who arrives at Mercer General ED following
witnessed cardiac arrest, resuscitated by paramedics en route. He is
haemodynamically unstable, unconscious, and requires immediate
defibrillation and advanced airway management.

The logistic regression classifies him as ESI 2 or ESI 3. Under ESI 2
he waits up to 10 minutes; under ESI 3 he waits up to 30 minutes. He
is directed to an acute cubicle, which has monitoring and intravenous
access but no defibrillator, no crash cart, and no advanced airway
equipment. Those are in the resuscitation bay. The tool that was
intended to support safer triage has directed him away from the only
space in the department where he can survive.

This is not a remote edge case. It is the most serious failure mode
in emergency triage — under-triaging a critically ill patient — and
it is precisely the failure the ESI protocol is designed to prevent.

### Why this failure occurs

ESI 1 is extremely rare in the training data. Vital sign patterns for
ESI 1 patients overlap substantially with ESI 2 (both present with
abnormal haemodynamics), making the boundary between these classes
difficult to learn from vital signs alone. Chief complaint language —
*cardiac arrest*, *unresponsive*, *major trauma*, *anaphylaxis* —
provides strongly discriminative signal that appears almost exclusively
in ESI 1 presentations. These terms are captured in the `cc_*` binary
columns, but the NLP layer that would process free-text chief complaints
and convert them to structured features has not yet been added.

### What the full model must address

The full stacked model must improve ESI 1 recall as its primary
performance target. This requires three additions beyond the baseline:

1. **Chief complaint NLP** — processing free-text triage complaint
   entries to extract ESI 1-associated terms and convert them to
   features the classifier can use.
2. **Synthetic minority oversampling (SMOTE)** or alternative class
   imbalance correction applied within each training fold to increase
   ESI 1 representation during training.
3. **AIF360 bias audit** — verifying that ESI 1 misclassification is
   not disproportionately worse for Afro-Caribbean patients, given
   the documented measurement bias in SpO2 readings for patients with
   darker skin pigmentation (Sjoding et al., 2020).

---

## References

- Sjoding MW, et al. Racial bias in pulse oximetry measurement.
  *NEJM*. 2020;383:2477–2478. doi:10.1056/NEJMc2029240
- Obermeyer Z, et al. Dissecting racial bias in an algorithm.
  *Science*. 2019;366:447–453. doi:10.1126/science.aax2342
- Tyler S, et al. Use of AI in Triage in Hospital EDs.
  *Cureus*. 2024;16(5). doi:10.7759/cureus.59906

