"""The single source of truth for what the ML model is and isn't allowed
to see. generate_dataset.py produces a long dataframe with everything —
the simulator's true latent state, the policy's label, and the noisy
observed values a person could actually check. This file draws the line
between them, so there is exactly one place that decision lives rather
than three places that could quietly drift apart.

TRUE_STATE_COLUMNS never reach the model. They exist to generate labels
and to let the week-ahead projection run the policy directly on simulated
future state. Feeding them to the classifier would mean the model is
approximating its own ground truth from the ground truth itself — the
leakage failure mode this project is explicitly built to avoid.
"""

import pandas as pd

LABEL_COLUMN = "label"

TRUE_STATE_COLUMNS = [
    "true_moisture",
    "true_stress_index",
]

CATEGORICAL_FEATURES = [
    "species",
    "age_cohort",
    "soil_profile",
]

NUMERIC_FEATURES = [
    "days_since_last_irrigation",
    "trailing_rainfall_7d_mm",
    "temp_max_c",
    "temp_min_c",
    "humidity_pct",
    "observed_moisture",
    "observed_stress_rating",
    "month",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Used only for the season-only sanity-check baseline in model.py, not part
# of the main feature set.
SEASON_ONLY_FEATURES = ["month"]


def assert_no_leakage():
    """Fails loudly (used in tests) if a true-state column has been added
    to the feature list by mistake.
    """
    overlap = set(FEATURE_COLUMNS) & set(TRUE_STATE_COLUMNS)
    if overlap:
        raise AssertionError(f"Leakage: true-state columns in features: {overlap}")
    if LABEL_COLUMN in FEATURE_COLUMNS:
        raise AssertionError("Leakage: label column is in the feature list")


def build_feature_matrix(df, feature_columns=None):
    """One-hot encode the categoricals, leave the numerics as-is. Returns
    (X, column_names) so callers can inspect what a trained model's
    coefficients/importances actually refer to.
    """
    cols = feature_columns or FEATURE_COLUMNS
    categorical = [c for c in cols if c in CATEGORICAL_FEATURES]
    numeric = [c for c in cols if c not in CATEGORICAL_FEATURES]

    X = df[numeric].copy()
    if categorical:
        dummies = pd.get_dummies(df[categorical], prefix=categorical)
        X = pd.concat([X, dummies], axis=1)
    return X, list(X.columns)
