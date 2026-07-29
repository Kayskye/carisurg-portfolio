# Handover Document
---

## (a) Project Summary

This project developed and evaluated a machine learning-assisted Emergency
Severity Index (ESI) triage tool for Mercer General Emergency Department,
a Caribbean regional hospital serving approximately 80,000 residents.
The tool is designed to augment — not replace — nurse-led triage decisions
by processing triage vital signs, arrival information, and presenting
complaints to generate a suggested ESI level (1–5), alongside a bias
audit that tests whether predictions are equitable across Afro-Caribbean
and Indo-Caribbean patient subgroups. Over eight weeks, the project
completed a structured literature review establishing the gap (no
published AI triage tool validated in a Caribbean ED), profiled the Yale
EMMLC triage dataset, trained and benchmarked nine classifiers, and
refactored the pipeline into a modular `src/` codebase driven by
`config.yaml`. The final selected model is LightGBM, chosen on the
grounds of balanced performance across all ESI classes and clinically
credible overall accuracy. The primary unresolved challenge — low recall
on ESI 1 (life-threatening) presentations — is the identified target for
Phase 2 improvement through chief complaint NLP and prospective validation
on real Mercer patient data.

---

## (b) Final Model Decision

**Selected model:** LightGBM (`lightgbm.LGBMClassifier`,
n_estimators=100, max_depth=10, learning_rate=0.1, class_weight=balanced,
random_state=42)

**One-sentence reason:** LightGBM achieved the highest Macro F1 (0.2848)
and an overall accuracy (0.4093) meaningfully above the stratified random
baseline (0.3754), demonstrating it has learned generalisable clinical
patterns from the training data — unlike the Logistic Regression, whose
accuracy of 0.2620 fell below the random baseline, indicating it
over-triages to an extent that would undermine clinical trust within a
single shift.

**Alternative considered and rejected:** Logistic Regression achieved
the highest ESI 1 recall (0.8125) but at the cost of overall accuracy
below the random baseline — a model that performs worse than chance on
overall accuracy would be abandoned by nursing staff regardless of its
performance on rare cases.

---

## (c) How to Run the Pipeline

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/[your-username]/carisurg-portfolio.git
cd carisurg-portfolio

# Install dependencies (all versions pinned)
pip install -r requirements.txt
```

### Place the dataset

The dataset is not committed to the repository (see Section d).
Place the CSV at the path specified in `config.yaml`:

```
data/yaleemmlc_admissionprediction_triage.csv
```

### Run end-to-end

```bash
python scripts/train.py --config config.yaml
```

This single command:
1. Loads and cleans the dataset (`src/data.py`)
2. Selects and preprocesses features (`src/features.py`)
3. Splits 80/20 stratified on ESI (random_seed from config)
4. Builds the LightGBM pipeline (`src/model.py`)
5. Trains, times, and evaluates the model
6. Saves metrics to `outputs/metrics.json` and model to `outputs/model.pkl`

### Module import test

```bash
python -c "import src.data; import src.features; import src.model; import src.utils; print('All modules OK')"
```

### Reproduce the Week 6–8 benchmarks

Run the relevant notebooks in `/notebooks/` — all use `random_state=42`
and the same 80/20 stratified split, so results are directly comparable.

---

## (d) Data Location and Governance

| Item | Detail |
|---|---|
| **Dataset name** | Yale EMMLC Admission Prediction Triage Dataset |
| **File** | `yaleemmlc_admissionprediction_triage.csv` |
| **Source** | Yale New Haven Health System (publicly released research dataset) |
| **Storage** | Local only — NOT committed to this repository |
| **Why not committed** | (1) File size — several hundred thousand rows; (2) Clinical data governance — even de-identified clinical datasets must not be shared via public repositories |
| **How to obtain** | Download from the original Yale EMMLC data release or the course data source |
| **`.gitignore` status** | `*.csv` pattern in `.gitignore` prevents accidental commit |
| **Governance note** | The dataset was collected under Yale institutional research governance. Any use at Mercer General ED requires a separate ethics approval covering Caribbean patient data collection and AI-assisted clinical decision support |

---

## (e) Known Limitations

**Limitation 1 — ESI 1 recall remains critically low.**
The final LightGBM model scores 0.0625 ESI 1 recall on the test set —
correctly identifying approximately 1 in 16 life-threatening patients.
This is insufficient for clinical deployment without the NLP chief
complaint layer, which is the primary planned improvement. Vital signs
alone do not provide enough discriminative signal to reliably separate
ESI 1 from ESI 2 presentations.

**Limitation 2 — No validation on real Mercer patient data.**
All results were produced on a simulated dataset derived from Yale EMMLC
data — not real Caribbean patient encounters. Phase 2 prospective data
collection at Mercer General ED is required before clinical conclusions
can be drawn. The model may perform differently on Caribbean presentations
(tropical disease profiles, sickle cell crisis, dengue fever) that are
absent from the Yale training data.

**Limitation 3 — SpO2 measurement bias unresolved.**
SpO2 is among the highest-importance features in the pipeline. Sjoding
et al. (2020) demonstrated that pulse oximeters systematically overestimate
oxygen saturation in patients with darker skin pigmentation. The AIF360
bias audit planned for Phase 2 will test demographic parity and equalised
odds, but cannot correct a measurement error at source.

**Limitation 4 — Hyperparameters not cross-validated.**
LightGBM hyperparameters in `config.yaml` (n_estimators=100, max_depth=10,
learning_rate=0.1) are principled starting points, not tuned values.
Cross-validated grid search is planned for Phase 2 and may yield
meaningful improvements on ESI 1 recall without requiring a more
complex model architecture.

**Limitation 5 — Indo-Caribbean demographic subgroup absent from training data.**
The Yale EMMLC dataset contains no Indo-Caribbean patients. The bias
audit cannot evaluate subgroups not represented in the data.
Phase 2 data collection must explicitly document ethnicity at triage
and ensure Indo-Caribbean presentations are captured.

---

## References

- Egerton-Warburton D, et al. *Emerg Med Australas*. 2025;37(4):e70086.
- Sjoding MW, et al. *NEJM*. 2020;383:2477–2478.
- Obermeyer Z, et al. *Science*. 2019;366:447–453.
- Coggan H, et al. *Pac Symp Biocomput*. 2026;31:551–565.
- Tyler S, et al. *Cureus*. 2024;16(5). doi:10.7759/cureus.59906

---

