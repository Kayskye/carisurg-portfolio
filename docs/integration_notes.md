---

## (a) What Data the Model Receives

**Input method: Manual entry by triage nurse**

All data is entered manually from the Mercer General ED paper triage
form. There is no EHR pull (no EHR exists) and no direct device
stream at this stage. The interface replicates the existing paper
form fields exactly — no new fields are added.

| Input field | Data type | Source | Notes |
|---|---|---|---|
| SpO2 | Numeric (%) | Nurse reads from pulse oximeter, enters manually | Flagged if 94–96% due to documented bias for Afro-Caribbean patients |
| Heart rate | Numeric (bpm) | Nurse reads from monitor, enters manually | |
| Systolic BP | Numeric (mmHg) | Nurse reads from BP cuff, enters manually | |
| Respiratory rate | Numeric (bpm) | Nurse counts, enters manually | |
| Temperature | Numeric (°C) | Nurse reads from thermometer, enters manually | |
| Age | Numeric (years) | From registration form, confirmed by nurse | |
| Blood glucose | Numeric (mmol/L) | Point-of-care device, entered manually | Missing for many patients — model handles with median imputation |
| Arrival mode | Categorical | Nurse selects from dropdown (Ambulance / Walk-in / Other) | |
| Chief complaint | Categorical + binary cc_ flags | Nurse selects from searchable list; maps to cc_ binary columns | |
| Previous disposition | Categorical | From patient history, entered if known | Optional — imputed if missing |

**What the model does not receive:**
- Free-text narrative notes (NLP layer not yet implemented)
- Lab results (not available at triage)
- Imaging (not available at triage)
- EHR data (no EHR exists at Mercer)

**Phase 2 planned additions:**
- Free-text chief complaint field processed by NLP layer
- Device stream from a validated pulse oximeter with skin-tone correction

---

## (b) What the Model Emits

The model produces three outputs simultaneously after the nurse
submits the triage form:

**Output 1 — ESI suggestion (primary)**
- Format: A single integer (1, 2, 3, 4, or 5)
- Display: Large colour-coded badge
  - ESI 1 → Red background, white text, pulsing border
  - ESI 2 → Orange background, white text
  - ESI 3 → Yellow background, dark text
  - ESI 4 → Green background, dark text
  - ESI 5 → Light grey background, dark text
- Timing: Displayed only after the nurse has recorded their own ESI
  level — never before, to prevent anchoring

**Output 2 — Confidence indicator**
- Format: Low / Medium / High label beneath the ESI badge
- Threshold: Suggestion is only shown if model confidence ≥ 80%.
  Below 80%, the badge is replaced with "Insufficient confidence —
  apply clinical judgment"

**Output 3 — SpO2 bias flag (conditional)**
- Triggered when: SpO2 input is in the 94–96% borderline range
- Display: Yellow warning banner above the ESI suggestion reading
  "SpO2 borderline — oximetry may underestimate hypoxaemia in
  patients with darker skin pigmentation. Apply clinical reassessment."
- This flag fires regardless of the model's ESI suggestion

**Output 4 — Deterioration alert (AI Plug-in 2, re-triage)**
- Triggered when: An ESI 4–5 patient has been waiting beyond the
  safe threshold without re-assessment, based on time elapsed and
  original ESI
- Display: Push notification to charge nurse tablet/screen
- Format: "Patient [ID] — waiting [X] minutes since ESI 4 triage.
  Visual re-check recommended."

---

## (c) What the Human Does Next

**After receiving the ESI suggestion:**

1. **If nurse's ESI matches the suggestion:**
   - Nurse taps "Confirm ESI [n]"
   - Both entries are logged (nurse's independent assessment AND
     the AI suggestion) — this creates the audit trail
   - Patient is directed to the appropriate zone
   - Time-stamp is recorded (triage complete)

2. **If nurse's ESI differs from the suggestion:**
   - Nurse taps "Override — my ESI is [n]"
   - Override is logged with the nurse's ESI, the model's ESI,
     and a timestamp — no written justification required
   - Patient is directed based on the nurse's decision
   - Override rate is tracked per shift for quality monitoring

3. **If the SpO2 bias flag is active:**
   - Nurse is prompted to consider a repeat SpO2 reading or
     apply clinical respiratory assessment
   - Nurse may escalate ESI independently of the model's suggestion
   - The flag does not block the workflow — the nurse can
     proceed without acting on it, but the flag is logged

4. **After the charge nurse receives a deterioration alert:**
   - Charge nurse conducts a visual re-check of the named patient
   - If deteriorated: charge nurse escalates ESI, moves patient
     to acute cubicle, documents re-triage event in the system
   - If stable: charge nurse acknowledges alert, documents check,
     sets next re-check timer
   - Alert is marked "acknowledged" only when a named clinician
     acts on it — it does not auto-dismiss

---

## Data Flow Diagram (Text)

```
NURSE enters vitals + chief complaint + arrival mode
        ↓
INTERFACE validates mandatory fields (flags missing data)
        ↓
MODEL receives structured input → runs LightGBM inference
        ↓
MODEL emits: ESI suggestion + confidence + SpO2 flag (if triggered)
        ↓
NURSE reviews suggestion (shown only after nurse's own ESI is recorded)
        ↓
   ┌────────────┬─────────────────────┐
   ↓            ↓                     ↓
CONFIRM      OVERRIDE             ESCALATE (ESI 1)
(log both)   (log nurse ESI,      (immediate resus
             model ESI,           bay — no tool
             timestamp)           interaction required)
        ↓
AUDIT LOG records: nurse ESI, model ESI, match/override, timestamp
        ↓
CHARGE NURSE receives shift summary at handover
        ↓
PHASE 2: data feeds retraining cycle → model improves over time
```

---

## Integration Constraints

- **Offline-first**: Inference must run locally (cached model) when
  internet connectivity is unavailable. Audit logs sync when
  connection is restored.
- **No EHR integration at this stage**: All inputs are manual.
  Phase 2 may introduce EHR pull if Mercer digitises records.
- **Session-based**: Each triage encounter is a separate session.
  The model has no memory of previous encounters for the same patient
  in this deployment version.
- **Data retention**: Audit logs retained for [X] months per Mercer
  governance policy. De-identified logs are used for model retraining.
