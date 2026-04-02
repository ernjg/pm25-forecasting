from pathlib import Path
from datetime import date, timedelta
import sys
import pandas as pd

# point to PM2.5 paper's code
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
BAMS_CODE = REPO_ROOT / "BAMS-PM25-Forecasting" / "code"

sys.path.append(str(BAMS_CODE))

from bams_pm25_forecast_assessment.daydataclass import DailyData

# build a starter dataset for LA-LB-A, using the same sources/time period as the PM2.5 paper
#LOCATION = "Los Angeles--Long Beach--Anaheim, CA Urban Area"
#START = date(2023, 8, 1)
#END = date(2023, 8, 14)   

#build a instead with longer time range (for sufficient datapoints)
LOCATION = "Los Angeles--Long Beach--Anaheim, CA Urban Area"
START = date(2023, 8, 1)
END = date(2023, 8, 30)  

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]

DATA_DIR = REPO_ROOT / "data" / "cache"
#DATA_DIR = Path("data/cache")

rows = []
d = START

while d <= END:
    day = DailyData(
        date=d,
        location_name=LOCATION,
        data_directory=DATA_DIR,
        _forecasts=["airnow", "geoscf"],
    )

    air = day.forecasts["airnow"].location_data.copy()
    geos = day.forecasts["geoscf"].location_data.copy()

    # average nearby points/monitors down to one hourly series each
    air_hourly = (
        air.groupby("ValidTime", as_index=False)["PM25"]
        .mean()
        .rename(columns={"PM25": "pm25_obs"})
    )

    geos_hourly = (
        geos.groupby("ValidTime", as_index=False)["PM25"]
        .mean()
        .rename(columns={"PM25": "pm25_geoscf"})
    )

    merged = air_hourly.merge(geos_hourly, on="ValidTime", how="inner")

    # 24-hour planning window:
    # forecasts initialized around 12 UTC, evaluated from 13 UTC same day
    # through 12 UTC next day -> ValidTime 13..36
    merged = merged[(merged["ValidTime"] >= 13) & (merged["ValidTime"] <= 36)].copy()

    merged["date"] = pd.Timestamp(d)
    merged["datehour"] = merged["date"] + pd.to_timedelta(merged["ValidTime"] - 1, unit="h")

    rows.append(merged[["datehour", "pm25_obs", "pm25_geoscf"]])
    d += timedelta(days=1)

df = pd.concat(rows, ignore_index=True).sort_values("datehour").reset_index(drop=True)

# cheap third feature
#df["pm25_obs_lag1"] = df["pm25_obs"].shift(1)

# add lag features for the previous 24 hours of observations 
for lag in range(1, 25):
    df[f"pm25_obs_lag{lag}"] = df["pm25_obs"].shift(lag)

#forecasrt lags 
df["pm25_geoscf_lag1"] = df["pm25_geoscf"].shift(1)
df["pm25_geoscf_lag3"] = df["pm25_geoscf"].shift(3)

#time features (day of week, hour)
df["hour"] = df["datehour"].dt.hour
df["dayofweek"] = df["datehour"].dt.dayofweek

# final starter dataset
df = df.dropna().reset_index(drop=True)
df.to_csv("data/los_angeles_starter_dataset.csv", index=False)

print(df.head())
print(df.shape)