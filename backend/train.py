# Trains plain LSTM + hybrid (SSA + SARIMAX + residual LSTM) for each city and writes metrics.json
# Usage: python train.py --source live|csv [--epochs 40]
import os, sys, json, random, argparse, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from common import *
from data_source import load_city

HERE = os.path.dirname(os.path.abspath(__file__))


def build_model(n_feat):
    m = models.Sequential([
        layers.Input((LOOK, n_feat)),
        layers.LSTM(64, return_sequences=True), layers.Dropout(0.2),
        layers.LSTM(32), layers.Dropout(0.2),
        layers.Dense(64, activation="relu"), layers.Dense(H)])
    m.compile(optimizer="adam", loss="mse")
    return m


def fit_lstm(Xtr, Ytr, Xva, Yva, epochs):
    m = build_model(Xtr.shape[2])
    es = callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    m.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=epochs, batch_size=64, callbacks=[es], verbose=2)
    return m


def rmse_mae(Y, P):
    out = {}
    for h in (6, 24, 72):
        e = Y[:, h - 1] - P[:, h - 1]
        out[str(h)] = dict(MAE=round(float(np.mean(np.abs(e))), 2), RMSE=round(float(np.sqrt(np.mean(e ** 2))), 2))
    return out


def main(source, epochs):
    out_dir = os.path.join(HERE, "..", "models", source)
    os.makedirs(out_dir, exist_ok=True)
    random.seed(0); np.random.seed(0); tf.random.set_seed(0)
    metrics = {}
    for city in CITY_SETS[source]:
        print("\n=====", city, "(%s data)" % source)
        d = load_city(source, city)
        n, idx = len(d), d.index
        print("hours:", n, "from", idx[0], "to", idx[-1])
        if n < 1200:
            print("Not enough data (need at least 1200 hours), skipping."); continue
        cut = int(n * 0.85)
        aqi_raw = d["AQI"].values
        mu, sd = d[COLS].iloc[:cut].mean().values, d[COLS].iloc[:cut].std().values
        F = make_features(d, mu, sd)
        U = fit_ssa(aqi_raw[:cut])
        S = ssa_smooth(aqi_raw, U); R = aqi_raw - S
        r_sd = np.nanstd(R[:cut])
        Fh = hybrid_features(F, R, r_sd)
        ok = ~np.isnan(Fh).any(axis=1)

        ok_tr = ok.copy(); ok_tr[cut:] = False
        a, b = longest_run(ok_tr)
        params = fit_sarimax(S[a:b], idx[a:b], ORDER)
        params_raw = fit_sarimax(aqi_raw[a:b], idx[a:b], ORDER_RAW, "c")

        cs = np.r_[0, np.cumsum(ok)]
        i = np.arange(LOOK, n - H + 1)
        i = i[cs[i + H] - cs[i - LOOK] == LOOK + H]
        tr, va = i[i + H <= cut], i[i >= cut]
        print("train/val samples:", len(tr), len(va))
        z = (aqi_raw - mu[0]) / sd[0]
        win = lambda A, o: np.stack([A[k - LOOK:k] for k in o])
        tgt = lambda A, o: np.stack([A[k:k + H] for k in o])

        print("-- plain LSTM")
        plain = fit_lstm(win(F, tr), tgt(z, tr), win(F, va), tgt(z, va), epochs)
        print("-- residual LSTM (hybrid)")
        hyb = fit_lstm(win(Fh, tr), tgt(R / r_sd, tr), win(Fh, va), tgt(R / r_sd, va), epochs)
        plain.save(os.path.join(out_dir, city + "_plain.keras"))
        hyb.save(os.path.join(out_dir, city + "_hybrid.keras"))
        np.savez(os.path.join(out_dir, city + ".npz"), mu=mu, sd=sd, r_sd=r_sd, U=U, params=params, params_raw=params_raw)

        stride = 6 if n < 4000 else 24
        vo = va[::stride]
        vo = vo[(vo >= CTX) & (cs[vo] - cs[vo - CTX] == CTX)]
        Y = tgt(aqi_raw, vo)
        t_last = lambda o: idx[o - 1]
        P = {}
        P["Persistence"] = np.stack([np.full(H, aqi_raw[o - 1]) for o in vo])
        P["SARIMAX"] = np.stack([sarimax_forecast(params_raw, aqi_raw[o - CTX:o], idx[o - CTX:o], t_last(o), ORDER_RAW, "c") for o in vo])
        SF = np.stack([sarimax_forecast(params, S[o - CTX:o], idx[o - CTX:o], t_last(o), ORDER) for o in vo])
        P["SSA + SARIMAX"] = SF
        P["Plain LSTM"] = plain.predict(win(F, vo), verbose=0) * sd[0] + mu[0]
        P["Hybrid (SSA+SARIMAX+LSTM)"] = SF + hyb.predict(win(Fh, vo), verbose=0) * r_sd
        res = {k: rmse_mae(Y, np.clip(v, 0, 500)) for k, v in P.items()}
        metrics[city] = dict(origins=int(len(vo)), val_start=str(idx[cut]), models=res)
        print("validation origins:", len(vo))
        for k, v in res.items():
            print("  %-26s RMSE +6/+24/+72:" % k, [v[h]["RMSE"] for h in ("6", "24", "72")])
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    print("\nDone. Models and metrics.json saved in", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="live", choices=["live", "csv"])
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    main(a.source, a.epochs)
