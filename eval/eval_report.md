# Evaluation report

Temporal, within-month split (see `model.py` docstring for why a plain
chronological tail-cut was rejected): 4,524 train rows, 1,092 test rows.

## Results

| Model | Macro F1 | Ordinal MAE |
|---|---|---|
| Majority class | 0.274 | 0.348 |
| Season-only (month alone) | 0.509 | 0.240 |
| Direct rule applied to observed features | 0.819 | 0.082 |
| **Random forest, full features** | **0.816** | **0.075** |
| Random forest, weather+block only (ablated) | 0.546 | 0.323 |

## What this shows

- **Not a calendar lookup.** Season-only reaches 0.509, nowhere near the
  full model's 0.816 — the model is reading actual conditions, not just
  knowing what month it is.
- **The model matches what a person applying the rule by hand to the same
  noisy readings would get.** The full model (0.816) and the direct rule
  applied to observed features (0.819) land within a hair of each other.
  This is the central experiment working as intended: a classical model
  can approximate the policy from imperfect observations about as well as
  mechanically applying the policy to those same observations would.
- **The soil-check and stress-rating features matter.** Dropping
  `observed_moisture`, `observed_stress_rating`, and
  `days_since_last_irrigation` drops macro F1 from 0.816 to 0.546 — the
  model genuinely depends on them, it isn't getting by on weather and
  species identity alone.

## Full-model classification report (test set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No irrigation | 0.991 | 0.965 | 0.977 | 761 |
| Monitor | 0.865 | 0.840 | 0.853 | 282 |
| Irrigate | 0.506 | 0.796 | 0.619 | 49 |

Confusion matrix (rows = true, columns = predicted; order None / Monitor / Irrigate):

```
[734,  27,   0]
[  7, 237,  38]
[  0,  10,  39]
```

The notable property: **zero two-step misses.** The model never predicts
"No irrigation" when the true label was "Irrigate," or the reverse. Every
error is a one-step confusion with the adjacent class. Precision on
Irrigate (0.506) is lower than recall (0.796) — the model over-flags
rather than under-flags, which is the safer direction for a water-stress
tool: a false "irrigate" costs some water and attention, a false "no
action needed" risks a tree actually going short.

## Top feature importances (full model)

| Feature | Importance |
|---|---|
| observed_moisture | 0.284 |
| observed_stress_rating | 0.204 |
| days_since_last_irrigation | 0.136 |
| trailing_rainfall_7d_mm | 0.106 |
| humidity_pct | 0.090 |
| temp_max_c | 0.039 |
| temp_min_c | 0.038 |
| month | 0.031 |

Month is the least important feature in the full model — further evidence
against the model leaning on the calendar.

## What this does and doesn't establish

- **Established:** the simulator and policy are implemented as documented
  and are reproducible from a fixed seed; the classifier approximates the
  authored policy well from noisy, realistically-available observations,
  on held-out data from the same single simulated cycle.
- **Not established:** that the policy itself reflects real orchard water
  needs, that these numbers would hold on a second year of real weather,
  or that this has been validated against actual irrigation outcomes. The
  labels are generated from an authored rule, not observed decisions.
