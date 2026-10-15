import pandas as pd
import pytest
import features


def test_no_leakage_by_default():
    features.assert_no_leakage()


def test_leakage_check_catches_true_state_in_features(monkeypatch):
    monkeypatch.setattr(features, "FEATURE_COLUMNS", features.FEATURE_COLUMNS + ["true_moisture"])
    with pytest.raises(AssertionError):
        features.assert_no_leakage()


def test_leakage_check_catches_label_in_features(monkeypatch):
    monkeypatch.setattr(features, "FEATURE_COLUMNS", features.FEATURE_COLUMNS + ["label"])
    with pytest.raises(AssertionError):
        features.assert_no_leakage()


def test_build_feature_matrix_one_hot_encodes_categoricals():
    df = pd.DataFrame({
        "species": ["coconut", "mango"],
        "age_cohort": ["mature", "young"],
        "soil_profile": ["medium", "fast_draining"],
        "days_since_last_irrigation": [3, 0],
        "trailing_rainfall_7d_mm": [10.0, 0.0],
        "temp_max_c": [32.0, 34.0],
        "temp_min_c": [20.0, 22.0],
        "humidity_pct": [60, 40],
        "observed_moisture": [100.0, 50.0],
        "observed_stress_rating": [1, 3],
        "month": [6, 3],
    })
    X, cols = features.build_feature_matrix(df)
    assert "species_coconut" in cols
    assert "age_cohort_young" in cols
    assert len(X) == 2
    assert "true_moisture" not in cols
