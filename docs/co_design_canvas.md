# Co-Design Canvas — Bias-Audited ESI Triage Support Tool
---

## Problem

**Clinical workflow being addressed:**
Triage nurses at Mercer General Emergency Department assign Emergency
Severity Index (ESI) levels 1–5 to every arriving patient during a
3–5 minute assessment window, using a paper intake form that cannot
be audited, trended, or searched. The nurse is the sole clinical
decision-maker at this step — no physician is present — and must
simultaneously capture vital signs (heart rate, blood pressure,
SpO2, respiratory rate, temperature, glucose), document the chief
complaint, record past medical history, and assign an acuity level,
all within the time constraint.

**The specific gap this tool addresses:**
The baseline ML model (LightGBM, Macro F1 = 0.2848) correctly
identifies only 6.25% of ESI 1 (life-threatening) patients.
A nurse using the tool's suggestion alone would miss approximately
15 of every 16 cardiac arrests, anaphylaxis presentations, and
major trauma cases. The tool is therefore positioned as a second
check — the nurse assigns their own ESI level first, the tool
provides a suggestion second, and the nurse either confirms or
overrides. The tool must not make the triage process slower,
and it must not require the nurse to enter any data beyond what
is already collected on the paper form.

**Who is affected:**
- Primary user: triage nurse (one per shift minimum at Mercer)
- Secondary user: charge nurse (receives deterioration alerts)
- Patient: receives correct or incorrect zone assignment based on ESI

---

## Ethics

**1. SpO2 measurement bias (Sjoding et al., 2020)**
Pulse oximeters systematically overestimate oxygen saturation in
patients with darker skin pigmentation — a well-documented hardware
bias directly relevant to Mercer's predominantly Afro-Caribbean
catchment. SpO2 is the highest-ranked feature in the model. If the
tool presents an ESI suggestion based on a falsely elevated SpO2
reading, it may recommend a lower acuity level than the patient
actually requires, compounding an existing measurement failure with
an algorithmic one. The interface must flag borderline SpO2 readings
(94–96%) explicitly and must not present the AI suggestion as
definitive when this flag is active.

**2. Automation bias on night shift**
When cognitive load is highest — at 03:00 with six patients waiting
— nurses are most likely to accept the tool's suggestion without
independent clinical evaluation. This converts a decision-support
tool into a decision-replacement tool. The interface must be designed
so the nurse records their own ESI level before the AI suggestion
is displayed. The suggestion must be visually secondary to the
nurse's own entry field.

**3. Informed consent for AI-assisted triage**
Patients presenting at Mercer in distress cannot meaningfully
consent to having their triage data processed by a machine learning
model. A notice must be displayed at the ED entrance, and an
opt-out mechanism must be available. This is documented in the
project risk register (Risk 4) and must be approved by the Mercer
ethics committee before deployment.

**4. Demographic underrepresentation in training data**
The model was trained on Yale EMMLC data (North American, no
Indo-Caribbean patients). The tool may perform differently for
Indo-Caribbean patients — approximately 35–40% of the population
in Trinidad and Tobago. The AIF360 bias audit will test demographic
parity across available subgroups, but cannot audit subgroups absent
from the training data. The interface must display a "Phase 2
validation pending" label until prospective Caribbean data confirms
equitable performance.

**5. ESI 1 false negatives — clinical harm pathway**
If the tool suggests ESI 3 for a patient in cardiac arrest (as the
baseline model currently does), the nurse may be influenced by the
suggestion to assign ESI 2 or 3 rather than ESI 1. The tool must
never suppress clinical escalation — if the nurse's independent
judgment is ESI 1, the tool must accept and record that override
without friction, and must not require the nurse to justify their
decision in a way that slows the resuscitation response.

---

## Environment

**Physical space:**
Mercer General ED operates from a single triage station adjacent to
the waiting room (~20 seats). The space is shared — the triage nurse
assesses patients at a counter or desk, with the patient or family
present. The device must be positioned so the AI suggestion is
visible to the nurse but not to the patient, to prevent patient
anxiety about algorithmic involvement in their care.

**Workflow constraints:**
- 3–5 minutes per triage assessment — the tool cannot add steps
  beyond what already exists on the paper form
- Minimum staffing: one triage nurse per shift; on heavy days
  (110–170 patients) the same nurse may triage continuously for
  several hours with no relief
- No separate paediatric triage — children are assessed at the
  same station with the same tool
- Re-triage is part of the workflow — the tool must capture
  re-assessment events, not just the initial ESI assignment

**Device and connectivity:**
- Target device: shared tablet (10–12 inch) at the triage station,
  or a low-specification desktop terminal
- Caribbean ED connectivity is intermittent — the tool must support
  offline inference using a locally cached model, with sync to a
  central server when connection is restored
- No integration with an EHR exists — all data is entered manually
  from the paper form; there is no EHR pull available at this stage

**Shift and handover:**
- Three shifts: day (08:00–20:00), evening overlap (14:00–22:00),
  night (20:00–08:00)
- The tool must generate a structured handover summary of
  outstanding alerts and re-triage flags at shift change
- Alerts must persist across shift boundaries until acknowledged
  by a named clinician

---

## Users and Stakeholders

| Stakeholder | Role in this tool | Primary concern |
|---|---|---|
| Triage nurse | Primary user — enters data, receives suggestion, confirms or overrides | Tool must not add to the 3–5 min assessment burden |
| Sister Patrice Alleyne (Nurse-in-Charge) | Clinical governance — decides whether tool is adopted | Must fit night shift reality; not abandoned under surge |
| Charge nurse | Receives deterioration alerts and handover summary | Real-time visibility of patient status and pending flags |
| ED physicians | Indirect beneficiary — cleaner triage data improves workup | No additional documentation burden |
| Patients and families | Affected by triage decision quality | Equitable care regardless of race or ethnic background |

---

## Features Required (Minimum Viable)

- **Vital signs entry form** — fields matching the existing Mercer paper triage form exactly; no new fields
- **Chief complaint selector** — searchable dropdown of common presenting complaints
- **ESI suggestion display** — colour-coded (ESI 1 red, 2 orange, 3 yellow, 4 green, 5 white) shown *after* nurse records their own ESI
- **SpO2 bias flag** — displayed when SpO2 is in the 94–96% borderline range
- **Override button** — one tap to record a different ESI; no justification required
- **Re-triage capture** — timestamps each re-assessment event for audit
- **Shift handover summary** — auto-generated at shift end listing outstanding alerts

---

## Success Criteria

| Criterion | Measurable indicator |
|---|---|
| Does not slow triage | Mean triage time ≤ baseline 3–5 min in pilot |
| Nurse engagement | Override rate 15–40% (too low = blind deference; too high = alert fatigue) |
| ESI 1 detection | ESI 1 recall ≥ 0.30 in Phase 2 prospective data |
| Equity | AIF360 demographic parity within 5% across Afro-Caribbean and Indo-Caribbean subgroups |
| Field completion | ≥ 95% of mandatory triage fields completed per encounter |

---

