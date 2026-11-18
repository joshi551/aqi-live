# Flask backend. Fetches fresh data on request (live mode) and forecasts with the saved models.
# Start:  python app.py            (live data)      or    AQI_SOURCE=csv python app.py
import os, json, time, numpy as np, tensorflow as tf
from flask import Flask, jsonify, request
from common import *
import data_source as ds

SOURCE = os.environ.get("AQI_SOURCE", "live")
HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.join(HERE, "..", "models", SOURCE)
app = Flask(__name__, static_folder=os.path.join(HERE, "..", "frontend"), static_url_path="")
CITIES = list(CITY_SETS[SOURCE])
fmt = lambda t: t.strftime("%Y-%m-%d %H:%M")

M = {}
for c in CITIES:
    p = os.path.join(MDIR, c)
    if os.path.exists(p + "_hybrid.keras"):
        M[c] = dict(st=np.load(p + ".npz"), plain=tf.keras.models.load_model(p + "_plain.keras"),
                    hyb=tf.keras.models.load_model(p + "_hybrid.keras"))

CACHE = {}


def get_df(city):
    c = CACHE.get(city)
    if c is None or (SOURCE == "live" and time.time() - c[0] > 900):     # live data re-fetched every 15 minutes
        CACHE[city] = (time.time(), ds.load_city(SOURCE, city))
    return CACHE[city][1]


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/api/info")
def info():
    note = ("Live hourly estimates from the Open-Meteo / CAMS air quality model, AQI computed with the CPCB method."
            if SOURCE == "live" else "Stored CPCB station data (ends Feb 2020), not live.")
    return jsonify(source=SOURCE, note=note, cities=CITIES, status=ds.STATUS)


@app.route("/api/refresh")
def refresh():
    CACHE.clear()
    return jsonify(ok=True)


@app.route("/api/metrics")
def metrics():
    p = os.path.join(MDIR, "metrics.json")
    return jsonify(json.load(open(p)) if os.path.exists(p) else {})


@app.route("/api/city/<city>")
def city_data(city):
    if city not in CITIES:
        return jsonify(error="unknown city"), 404
    try:
        d = get_df(city)
    except Exception as e:
        return jsonify(error=str(e)), 502
    hours = min(int(request.args.get("hours", 168)), 720)
    last = d["AQI"].last_valid_index()
    w = d.loc[last - pd.Timedelta(hours=hours - 1):last]
    latest = {p: (None if w[p].iloc[-6:].dropna().empty else round(float(w[p].iloc[-6:].dropna().iloc[-1]), 1)) for p in POLL}
    return jsonify(city=city, as_of=fmt(last), times=[fmt(t) for t in w.index],
                   aqi=[None if np.isnan(v) else int(round(v)) for v in w["AQI"]], pollutants=latest)


@app.route("/api/forecast/<city>")
def forecast(city):
    if city not in M:
        return jsonify(error="models not trained for %s data, run: python train.py --source %s" % (SOURCE, SOURCE)), 404
    try:
        d = get_df(city)
    except Exception as e:
        return jsonify(error=str(e)), 502
    m = M[city]; st = m["st"]
    aqi = d["AQI"].values
    F = make_features(d, st["mu"], st["sd"])
    S = ssa_smooth(aqi, st["U"])
    Fh = hybrid_features(F, aqi - S, float(st["r_sd"]))
    ok = ~np.isnan(Fh).any(axis=1)
    cs = np.r_[0, np.cumsum(ok)]
    e = len(d)
    while e >= CTX and cs[e] - cs[e - LOOK] != LOOK:       # latest 72-hour window with no missing hour
        e -= 1
    if e < CTX:
        return jsonify(error="no complete input window in the latest data"), 404
    t0 = d.index[e - 1]
    sf = sarimax_forecast(st["params"], S[e - CTX:e], d.index[e - CTX:e], t0, ORDER)
    hyb = np.clip(sf + m["hyb"].predict(Fh[e - LOOK:e][None], verbose=0)[0] * float(st["r_sd"]), 0, 500)
    pl = np.clip(m["plain"].predict(F[e - LOOK:e][None], verbose=0)[0] * st["sd"][0] + st["mu"][0], 0, 500)
    times = [fmt(t0 + pd.Timedelta(hours=k)) for k in range(1, H + 1)]
    return jsonify(city=city, origin=fmt(t0), times=times, aqi=[int(round(v)) for v in hyb],
                   lstm=[int(round(v)) for v in pl], horizons={str(h): int(round(hyb[h - 1])) for h in (6, 24, 72)})


if __name__ == "__main__":
    app.run(port=5000, debug=False)
