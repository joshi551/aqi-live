# Data and forecast helpers used by the Streamlit app (same logic as the Flask API in app.py)
import os, numpy as np, pandas as pd
from common import *

fmt = lambda t: t.strftime("%Y-%m-%d %H:%M")


def load_models(models_dir, cities):
    import tensorflow as tf
    out = {}
    for c in cities:
        p = os.path.join(models_dir, c)
        if os.path.exists(p + "_hybrid.keras"):
            out[c] = dict(st=dict(np.load(p + ".npz")), plain=tf.keras.models.load_model(p + "_plain.keras"),
                          hyb=tf.keras.models.load_model(p + "_hybrid.keras"))
    return out


def city_payload(city, d, hours=168):
    last = d["AQI"].last_valid_index()
    w = d.loc[last - pd.Timedelta(hours=hours - 1):last]
    latest = {p: (None if w[p].iloc[-6:].dropna().empty else round(float(w[p].iloc[-6:].dropna().iloc[-1]), 1)) for p in POLL}
    return dict(city=city, as_of=fmt(last), times=[fmt(t) for t in w.index],
                aqi=[None if np.isnan(v) else int(round(v)) for v in w["AQI"]], pollutants=latest)


def forecast_payload(city, d, m):
    if m is None:
        return {"__err": "models not trained for this data source, run backend/train.py first"}
    st = m["st"]
    aqi = d["AQI"].values
    F = make_features(d, st["mu"], st["sd"])
    S = ssa_smooth(aqi, st["U"])
    Fh = hybrid_features(F, aqi - S, float(st["r_sd"]))
    ok = ~np.isnan(Fh).any(axis=1)
    cs = np.r_[0, np.cumsum(ok)]
    e = len(d)
    while e >= CTX and cs[e] - cs[e - LOOK] != LOOK:
        e -= 1
    if e < CTX:
        return {"__err": "no complete input window in the latest data"}
    t0 = d.index[e - 1]
    sf = sarimax_forecast(st["params"], S[e - CTX:e], d.index[e - CTX:e], t0, ORDER)
    hyb = np.clip(sf + m["hyb"].predict(Fh[e - LOOK:e][None], verbose=0)[0] * float(st["r_sd"]), 0, 500)
    pl = np.clip(m["plain"].predict(F[e - LOOK:e][None], verbose=0)[0] * st["sd"][0] + st["mu"][0], 0, 500)
    return dict(city=city, origin=fmt(t0), times=[fmt(t0 + pd.Timedelta(hours=k)) for k in range(1, H + 1)],
                aqi=[int(round(v)) for v in hyb], lstm=[int(round(v)) for v in pl],
                horizons={str(h): int(round(hyb[h - 1])) for h in (6, 24, 72)})
