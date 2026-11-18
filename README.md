Real-time AQI forecasting: SSA + SARIMA + LSTM hybrid, 2 cities
Two ways to show it: Flask website (local) or Streamlit app (free live link)

DATA MODES
  live (default)  Open-Meteo Air Quality API, hourly pollutants, any latitude/longitude, no key.
                  Cities: Visakhapatnam, Vijayawada. AQI calculated with the CPCB method. CAMS model estimates, not station data.
  csv             CPCB station file city_hour.csv (to Feb 2020). Cities: Visakhapatnam, Amaravati.

FILES
  backend/common.py       settings, SSA, SARIMA helpers
  backend/data_source.py  live download, CPCB AQI calculation, csv loading
  backend/train.py        trains plain LSTM + hybrid per city, writes models/<mode>/metrics.json
  backend/service.py      data and forecast helpers used by the Streamlit app
  backend/app.py          Flask server and API (local website)
  backend/collect.py      downloads new data only (schedule it daily)
  frontend/index.html     the dashboard page (used by both Flask and Streamlit)
  streamlit_app.py        Streamlit app (main file for Streamlit Community Cloud)
  .streamlit/config.toml  Streamlit theme

A. RUN LOCALLY
  1. python -m pip install -r requirements.txt
  2. cd backend
  3. python train.py --source live        (needs internet; --epochs 3 for a quick test)
  4. Flask site:     python app.py                      then open http://127.0.0.1:5000
     Streamlit site: go back to the project folder, then run: python -m streamlit run streamlit_app.py
  CSV mode: put city_hour.csv in data/, python train.py --source csv, then set AQI_SOURCE=csv
     (CMD: set AQI_SOURCE=csv   PowerShell: $env:AQI_SOURCE="csv") before starting either site.

B. PUT IT ON A LIVE LINK (Streamlit Community Cloud, free)
  1. Train locally first (step A3). Check that models/live has .keras and .npz files and metrics.json.
  2. Create a PUBLIC repository on github.com and upload this whole folder, including models/live.
     (data/live and data/city_hour.csv are left out by .gitignore.)
  3. Go to share.streamlit.io, sign in with GitHub, click Create app.
  4. Pick the repository, branch main, main file path streamlit_app.py.
  5. Open Advanced settings and choose Python 3.11 (TensorFlow does not support the newest Python versions).
  6. Click Deploy. The first build takes several minutes. Then open the link.
  Notes: the app sleeps when unused and wakes when someone opens the link. Live data is downloaded on each visit (cached 15 minutes).
  If the build says out of memory, TensorFlow is too big for the free plan; use the lighter TFLite version.

HOW IT WORKS
  SSA (window 24, 3 components) -> smooth part + residual. SARIMA (2,1,1) + daily Fourier terms forecasts the smooth part.
  LSTM (72 h in, 72 h out) forecasts the residual. Forecast = SARIMA forecast + LSTM forecast, limited to 0 to 500.
  train.py prints and saves validation errors for Persistence, SARIMA, SSA+SARIMA, plain LSTM and the hybrid.

HONEST NOTES
  Read the validation table before claiming the hybrid is best; it may not be.
  The same last 15% of the data is used for early stopping and for the reported errors, so the LSTM numbers are slightly optimistic.
  Live mode starts with about 92 days of history, so the LSTM has little data. Let history grow with collect.py, or use csv mode.
