"""Ties simulate.py (physics) and policy.py (decision) together into the
labeled dataset: one row per block per day. This is the only place the two
are coupled — irrigation decided by the policy today changes tomorrow's
moisture, which is exactly how the real orchard works (water applied today
changes what's needed next week).

Burn-in: the first BURN_IN_DAYS of each block's simulation are discarded.
They exist only to let the initial moisture assumption and the 7-day
rolling rainfall window settle into something the weather data actually
drove, not an arbitrary starting guess.

Run: python generate_dataset.py
Produces data/orchard_dataset.csv, checked into the repo so the labeled
data is inspectable and reproducible without rerunning the simulation.
"""

import numpy as np
import pandas as pd

import simulate
import policy
from blocks import BLOCKS, block_params

SEED = 42
BURN_IN_DAYS = 20
INITIAL_FILL_FRACTION = 0.70  # starting moisture as a fraction of root capacity


def load_weather():
    df = pd.read_csv("data/weather_history.csv", parse_dates=["date"])
    df["rainfall_7d_mm"] = df["rainfall_mm"].rolling(7, min_periods=1).sum()
    df["month"] = df["date"].dt.month
    return df


def simulate_block(block, weather, rng):
    params = block_params(block)
    root_cap = params["root_capacity_mm"]

    moisture = root_cap * INITIAL_FILL_FRACTION
    stress_prev = simulate.water_stress_index(moisture, root_cap, params["stress_sensitivity"])
    days_since_irrigation = 5  # arbitrary plausible starting point, washed out by burn-in

    rows = []
    prev_rainfall = 0.0

    for t, day in weather.iterrows():
        rainfall = day["rainfall_mm"]
        et0 = day["et0_mm"]

        moisture_natural = simulate.step_moisture(
            moisture, rainfall, et0,
            params["water_demand_coefficient"], root_cap, params["drainage_rate"],
        )
        stress_true = simulate.water_stress_index(moisture_natural, root_cap, params["stress_sensitivity"])
        recent_rain_2d = rainfall + prev_rainfall

        label = policy.determine_irrigation_need(stress_true, recent_rain_2d)

        if label == policy.IRRIGATE:
            irrig_mm = policy.irrigation_amount_mm(moisture_natural, root_cap)
            moisture_final = simulate.step_moisture(
                moisture_natural, rainfall_mm=0, et0_mm=0,
                water_demand_coefficient=0, root_capacity_mm=root_cap,
                drainage_rate=params["drainage_rate"], irrigation_mm=irrig_mm,
            )
        else:
            moisture_final = moisture_natural

        stress_rating = simulate.visible_stress_rating(stress_true, stress_prev, rng)
        obs_moisture = simulate.observed_moisture(moisture_natural, root_cap, rng)

        if t >= BURN_IN_DAYS:
            rows.append({
                "date": day["date"],
                "block_id": block["id"],
                "species": block["species"],
                "age_cohort": block["age_cohort"],
                "soil_profile": block["soil_profile"],
                "is_commercial": params["is_commercial"],
                "true_moisture": moisture_natural,
                "true_stress_index": stress_true,
                "observed_moisture": obs_moisture,
                "observed_stress_rating": stress_rating,
                "days_since_last_irrigation": days_since_irrigation,
                "trailing_rainfall_7d_mm": day["rainfall_7d_mm"],
                "temp_max_c": day["temp_max_c"],
                "temp_min_c": day["temp_min_c"],
                "humidity_pct": day["humidity_pct"],
                "month": day["month"],
                "label": label,
            })

        moisture = moisture_final
        stress_prev = stress_true
        days_since_irrigation = 0 if label == policy.IRRIGATE else days_since_irrigation + 1
        prev_rainfall = rainfall

    return rows


def generate():
    weather = load_weather()
    rng = np.random.default_rng(SEED)
    all_rows = []
    for block in BLOCKS:
        all_rows.extend(simulate_block(block, weather, rng))
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/orchard_dataset.csv", index=False)
    print(f"Generated {len(df)} rows across {df['block_id'].nunique()} blocks")
    print(df["label"].map(policy.LABELS).value_counts())
    print(df.groupby("species")["label"].mean().sort_values())
