# Data sources: "live" (Open-Meteo Air Quality API, hourly, no key) and "csv" (CPCB city_hour.csv)
import os, time, numpy as np, pandas as pd, requests
from common import *

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "data", "live")
CSV_PATH = os.path.join(HERE, "..", "data", "city_hour.csv")
URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
API_VARS = {"pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO2",
            "sulphur_dioxide": "SO2", "carbon_monoxide": "CO", "ozone": "O3"}
STATUS = {}          # per city: where the latest data came from

# CPCB breakpoints (concentration -> index 0,50,100,200,300,400,500); CO in mg/m3
IDX = [0, 50, 100, 200, 300, 400, 500]
BP = {"PM2.5": [0, 30, 60, 90, 120, 250, 380], "PM10": [0, 50, 100, 250, 350, 430, 510],
      "NO2": [0, 40, 80, 180, 280, 400, 800], "SO2": [0, 40, 80, 380, 800, 1600, 2000],
      "CO": [0, 1, 2, 10, 17, 34, 50], "O3": [0, 50, 100, 168, 208, 748, 1000]}


def cpcb_aqi(p):
    """AQI from hourly pollutants: 24h mean for PM2.5, PM10, NO2, SO2; 8h mean for CO, O3."""
    avg = pd.DataFrame(index=p.index)
    for k in ["PM2.5", "PM10", "NO2", "SO2"]:
        avg[k] = p[k].rolling(24, min_periods=16).mean()
    for k in ["CO", "O3"]:
        avg[k] = p[k].rolling(8, min_periods=6).mean()
    sub = pd.DataFrame({k: np.interp(avg[k], BP[k], IDX) for k in BP}, index=p.index).where(avg.notna())
    return sub.max(axis=1).where(avg.notna().sum(axis=1) >= 3)


def fetch_live(lat, lon):
    r = requests.get(URL, params=dict(latitude=lat, longitude=lon, hourly=",".join(API_VARS),
                                      past_days=92, forecast_days=1, timezone="Asia/Kolkata"), timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({API_VARS[k]: pd.to_numeric(pd.Series(h[k]), errors="coerce") for k in API_VARS})
    df.index = pd.to_datetime(h["time"]); df.index.name = "Datetime"
    df["CO"] = df["CO"] / 1000.0                            # ug/m3 -> mg/m3
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("h")
    return df.loc[:now]                                     # drop hours in the future


def finish(df, aqi=None):
    d = df.asfreq("h")
    d["AQI"] = cpcb_aqi(d) if aqi is None else aqi.reindex(d.index)
    d = d[COLS]
    for c in COLS:
        d[c] = fill_short(d[c], MAX_FILL)
    first = d.dropna(how="all").index[0]
    return d.loc[first:]


def load_live(city):
    lat, lon = CITY_SETS["live"][city]
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, city + ".csv")
    old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else None
    try:
        new = fetch_live(lat, lon)
        df = new if old is None else pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()      # history keeps growing
        df.to_csv(path)
        STATUS[city] = dict(live=True, fetched=time.strftime("%Y-%m-%d %H:%M:%S"), note="fresh from Open-Meteo")
    except Exception as e:
        if old is None:
            raise RuntimeError("Could not download live data and there is no saved copy: %s" % e)
        df = old
        STATUS[city] = dict(live=False, fetched=time.strftime("%Y-%m-%d %H:%M:%S"),
                            note="download failed, using saved copy (%s)" % type(e).__name__)
    return finish(df)


_CSV = {}


def load_csv(city):
    if not _CSV:
        _CSV["raw"] = pd.read_csv(CSV_PATH, parse_dates=["Datetime"], usecols=["City", "Datetime", "AQI"] + POLL)
    d = _CSV["raw"]
    d = d[d.City == city].set_index("Datetime").sort_index()
    d = d[~d.index.duplicated()].loc[:CSV_END]
    STATUS[city] = dict(live=False, fetched=time.strftime("%Y-%m-%d %H:%M:%S"), note="stored CPCB data (to Feb 2020)")
    return finish(d[POLL], aqi=d["AQI"])


def load_city(source, city):
    return load_live(city) if source == "live" else load_csv(city)
