"""The week-ahead projection: reuses simulate.py + policy.py exactly as
generate_dataset.py does, but driven by a live 7-day weather forecast
instead of historical weather, and seeded from a manually-entered current
condition instead of a simulated history.

This deliberately does NOT call the trained classifier (model.py). There
is no live sensor feed in this demo — no real observed_moisture or
observed_stress_rating to hand it — so running the classifier here would
mean feeding it placeholder numbers dressed up as sensor data. The
classifier's accuracy claim is validated on the historical test set
instead (see eval/eval_report.md and the Model Details tab); this module
answers a different, honestly-scoped question: "if conditions are roughly
X today, how does the deterministic policy expect things to evolve over
the week, given the real forecast?"

Confidence is not a model output here — it's mechanical: day 0 rests on
today's forecast (usually solid), day 6 rests on a week-out forecast
compounded through six days of simulated water balance. The app expresses
that as declining visual certainty, not a number this module invents.
"""

import requests
import pandas as pd

import simulate
import policy
from blocks import BLOCKS, block_params

NASHIK_LAT = 20.0059
NASHIK_LON = 73.7910

STARTING_CONDITIONS = {
    "Just watered / recent rain": 0.90,
    "Normal": 0.70,
    "Getting dry": 0.50,
    "Very dry": 0.30,
}


def fetch_forecast_weather(forecast_days=7):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": NASHIK_LAT,
        "longitude": NASHIK_LON,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "relative_humidity_2m_mean,et0_fao_evapotranspiration",
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    daily = resp.json()["daily"]
    df = pd.DataFrame(daily).rename(columns={
        "time": "date",
        "temperature_2m_max": "temp_max_c",
        "temperature_2m_min": "temp_min_c",
        "precipitation_sum": "rainfall_mm",
        "relative_humidity_2m_mean": "humidity_pct",
        "et0_fao_evapotranspiration": "et0_mm",
    })
    df["date"] = pd.to_datetime(df["date"])
    return df


def project_block(block, forecast, starting_fill_fraction):
    params = block_params(block)
    root_cap = params["root_capacity_mm"]

    moisture = root_cap * starting_fill_fraction
    stress_prev = simulate.water_stress_index(moisture, root_cap, params["stress_sensitivity"])
    prev_rainfall = 0.0  # unknown day before the forecast window starts; a stated simplification

    rows = []
    for day_index, (_, day) in enumerate(forecast.iterrows()):
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
                moisture_natural, rainfall_mm=0, et0_mm=0, water_demand_coefficient=0,
                root_capacity_mm=root_cap, drainage_rate=params["drainage_rate"], irrigation_mm=irrig_mm,
            )
        else:
            moisture_final = moisture_natural

        rows.append({
            "date": day["date"],
            "day_index": day_index,
            "block_id": block["id"],
            "species": block["species"],
            "label": label,
            "label_name": policy.LABELS[label],
            "stress_index": stress_true,
            "rainfall_mm": rainfall,
        })

        moisture = moisture_final
        stress_prev = stress_true
        prev_rainfall = rainfall

    return rows


def project_all_blocks(starting_condition="Normal", forecast_days=7):
    fill_fraction = STARTING_CONDITIONS[starting_condition]
    forecast = fetch_forecast_weather(forecast_days)
    all_rows = []
    for block in BLOCKS:
        all_rows.extend(project_block(block, forecast, fill_fraction))
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = project_all_blocks("Normal")
    for block_id, group in df.groupby("block_id"):
        timeline = " ".join(f"{row.label_name[0]}" for row in group.itertuples())
        print(f"{block_id:4s} {timeline}")
