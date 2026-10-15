"""One-off script: pull real historical daily weather for Nashik from
Open-Meteo's free archive API (no key needed) and cache it as a CSV.

Run once: python fetch_weather.py
Produces data/weather_history.csv, which is checked into the repo so the
dataset is reproducible without hitting the network again.
"""

import requests
import pandas as pd

NASHIK_LAT = 20.0059
NASHIK_LON = 73.7910

START_DATE = "2023-06-01"
END_DATE = "2024-09-30"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "et0_fao_evapotranspiration",
]


def fetch():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": NASHIK_LAT,
        "longitude": NASHIK_LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["daily"]
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "time": "date",
        "temperature_2m_max": "temp_max_c",
        "temperature_2m_min": "temp_min_c",
        "precipitation_sum": "rainfall_mm",
        "relative_humidity_2m_mean": "humidity_pct",
        "et0_fao_evapotranspiration": "et0_mm",
    })
    return df


if __name__ == "__main__":
    df = fetch()
    df.to_csv("data/weather_history.csv", index=False)
    print(f"Saved {len(df)} days to data/weather_history.csv")
    print(df.head())
    print(df.tail())
