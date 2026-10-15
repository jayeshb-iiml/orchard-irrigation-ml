import numpy as np
import pandas as pd
import pytest

from blocks import BLOCKS
from generate_dataset import simulate_block
import policy


def _toy_weather(n_days=40):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=n_days)
    rainfall = rng.choice([0, 0, 0, 5, 20], size=n_days)
    return pd.DataFrame({
        "date": dates,
        "rainfall_mm": rainfall,
        "et0_mm": rng.uniform(3, 7, size=n_days),
        "rainfall_7d_mm": pd.Series(rainfall).rolling(7, min_periods=1).sum(),
        "month": dates.month,
        "temp_max_c": rng.uniform(25, 38, size=n_days),
        "temp_min_c": rng.uniform(15, 25, size=n_days),
        "humidity_pct": rng.uniform(30, 90, size=n_days),
    })


def test_simulation_is_reproducible_given_the_same_seed():
    weather = _toy_weather()
    block = BLOCKS[0]
    rows_a = simulate_block(block, weather, np.random.default_rng(7))
    rows_b = simulate_block(block, weather, np.random.default_rng(7))
    assert rows_a == rows_b


def test_simulation_produces_different_noise_with_different_seeds():
    weather = _toy_weather()
    block = BLOCKS[0]
    rows_a = simulate_block(block, weather, np.random.default_rng(1))
    rows_b = simulate_block(block, weather, np.random.default_rng(2))
    moistures_a = [r["observed_moisture"] for r in rows_a]
    moistures_b = [r["observed_moisture"] for r in rows_b]
    assert moistures_a != moistures_b


def test_true_moisture_never_negative():
    weather = _toy_weather(n_days=60)
    for block in BLOCKS:
        rows = simulate_block(block, weather, np.random.default_rng(3))
        assert all(r["true_moisture"] >= 0 for r in rows)


def test_labels_are_always_valid():
    weather = _toy_weather(n_days=60)
    for block in BLOCKS:
        rows = simulate_block(block, weather, np.random.default_rng(3))
        assert all(r["label"] in (policy.NONE, policy.MONITOR, policy.IRRIGATE) for r in rows)


def test_days_since_last_irrigation_resets_after_an_irrigate_day():
    weather = _toy_weather(n_days=60)
    block = BLOCKS[0]
    rows = simulate_block(block, weather, np.random.default_rng(3))
    for i in range(1, len(rows)):
        if rows[i - 1]["label"] == policy.IRRIGATE:
            assert rows[i]["days_since_last_irrigation"] == 0


@pytest.mark.parametrize("path", ["data/orchard_dataset.csv"])
def test_checked_in_dataset_has_no_leakage_columns_missing(path):
    df = pd.read_csv(path)
    required = {"true_moisture", "true_stress_index", "observed_moisture",
                "observed_stress_rating", "label", "block_id", "date"}
    assert required.issubset(df.columns)


def test_checked_in_dataset_moisture_is_never_negative():
    df = pd.read_csv("data/orchard_dataset.csv")
    assert (df["true_moisture"] >= 0).all()


def test_checked_in_dataset_has_all_three_labels_represented():
    df = pd.read_csv("data/orchard_dataset.csv")
    assert set(df["label"].unique()) == {0, 1, 2}


def test_checked_in_dataset_rows_per_block_are_consistent():
    df = pd.read_csv("data/orchard_dataset.csv")
    counts = df.groupby("block_id").size()
    assert counts.nunique() == 1
