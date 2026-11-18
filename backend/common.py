# Shared settings and the SSA / SARIMAX helpers
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

POLL = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
COLS = ["AQI"] + POLL
NF = len(COLS)
LOOK = 72             # hours of history for the LSTM
H = 72                # hours forecast
CTX = 168             # hours of history for SARIMAX
MAX_FILL = 24         # only gaps up to 24 hours are filled
SSA_L, SSA_K = 24, 3  # SSA window length, components kept
ORDER = (2, 1, 1)     # SARIMAX order for the SSA smooth part
ORDER_RAW = (2, 0, 1) # SARIMAX order for raw AQI (with constant)

# live mode: any city by latitude / longitude. csv mode: cities found in city_hour.csv
CITY_SETS = {
    "live": {"Visakhapatnam": (17.6868, 83.2185), "Vijayawada": (16.5062, 80.6480)},
    "csv": {"Visakhapatnam": None, "Amaravati": None},
}
CSV_END = "2020-02-29 23:00"      # csv mode stops before the COVID lockdown


def fill_short(s, maxgap):
    na = s.isna()
    grp = (na != na.shift()).cumsum()
    run = na.groupby(grp).transform("sum")
    filled = s.interpolate(limit_area="inside")
    out = s.copy()
    m = na & (run <= maxgap) & filled.notna()
    out[m] = filled[m]
    return out


def make_features(d, mu, sd):
    z = (d[COLS].values - mu) / sd
    hr = d.index.hour.values
    return np.column_stack([z, np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)])


def hybrid_features(F, R, r_sd):
    return np.column_stack([F[:, :NF], R / r_sd, F[:, NF:]])   # features + SSA residual + hour sin/cos


def fit_ssa(x, L=SSA_L, K=SSA_K):
    W = np.lib.stride_tricks.sliding_window_view(x, L)
    W = W[~np.isnan(W).any(axis=1)]
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    return Vt[:K].T


def ssa_smooth(x, U):
    L = U.shape[0]
    P = np.lib.stride_tricks.sliding_window_view(x, L) @ U @ U.T
    S = np.full(len(x), np.nan)
    S[L - 1:] = P[:, -1]          # uses only the previous L hours (no future data)
    return S


def fourier(idx, K=2):
    h = np.asarray(idx.hour, dtype=float)
    cols = []
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * h / 24), np.cos(2 * np.pi * k * h / 24)]
    return np.column_stack(cols)


def longest_run(ok, max_len=3000):
    best, s = (0, 0), None
    for i, v in enumerate(list(ok) + [False]):
        if v and s is None:
            s = i
        if not v and s is not None:
            if i - s > best[1] - best[0]:
                best = (s, i)
            s = None
    a, b = best
    return max(a, b - max_len), b


def fit_sarimax(series, idx, order, trend=None):
    return np.asarray(SARIMAX(series, exog=fourier(idx), order=order, trend=trend).fit(disp=False, maxiter=100).params)


def sarimax_forecast(params, window, idx_window, t_last, order, trend=None):
    idx_future = pd.date_range(t_last + pd.Timedelta(hours=1), periods=H, freq="h")
    r = SARIMAX(window, exog=fourier(idx_window), order=order, trend=trend).smooth(params)
    return np.asarray(r.forecast(H, exog=fourier(idx_future)))
