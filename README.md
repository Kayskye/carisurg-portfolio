# ESI Triage Augmentation Tool — Mercer General Emergency Department

A bias-audited machine learning tool designed to support nurse-led ESI triage decisions at Mercer General ED, a resource-constrained Caribbean regional hospital. The tool augments — not replaces — the existing five-level ESI workflow, flagging potential acuity miscategorisations in real time to improve patient flow and reduce ED overcrowding.

---

## What Is This Project?

This project is a 12-week MedPath feasibility study developing an ML-augmented ESI triage decision support system for Mercer General ED. Two compounding problems motivate the work:

1. **Documentation gap** — Mercer's paper triage forms cannot be audited or trended, leaving miscategorisation undetectable across 70–95 daily encounters.
2. **Demographic bias** — Existing AI triage tools underestimate illness severity in racially diverse Caribbean populations due to demographic measurement bias.

The tool addresses both by digitising triage data capture and auditing model outputs for fairness across demographic groups.

---

## Who Is This For?

| User | Role |
|---|---|
| Triage nurses | Primary users — ESI workflow unchanged; tool provides a background second check |
| Nurse-in-charge | Reviews flagged miscategorisations and re-triage alerts |
| MedPath assessors | Evaluating feasibility, bias audit methodology, and clinical relevance |

> This tool is validated against simulated Mercer General ED patient cases. It is not yet cleared for live clinical deployment.

---

## Project Structure

```
carisurg-portfolio/
├── notebooks/        # Exploratory Jupyter notebooks — triage data analysis and model prototyping
├── docs/             # Memos, proposals, and reports — includes problem statement and pilot design
├── data/             # Raw and cleaned simulated Caribbean triage cases
├── src/              # Reusable modules (Week 8+) — NLP pipeline, ML classifier, bias audit
├── README.md         # The front door
├── LICENSE           # How others can use this project
├── .gitignore        # What Git never tracks
└── requirements.txt  # Exact library versions
```
