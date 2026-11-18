# Streamlit version of the AQI forecast dashboard.
# Run locally:   streamlit run streamlit_app.py
# Data source:   live (default) or csv  ->  set the AQI_SOURCE environment variable, or a secret named AQI_SOURCE
import os, sys, json, time
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend"))
from common import CITY_SETS
import data_source as ds
import service

try:
    SOURCE = st.secrets.get("AQI_SOURCE", os.environ.get("AQI_SOURCE", "live"))
except Exception:
    SOURCE = os.environ.get("AQI_SOURCE", "live")
CITIES = list(CITY_SETS[SOURCE])
MDIR = os.path.join(ROOT, "models", SOURCE)

st.set_page_config(page_title="AQI Forecast", page_icon="🌫", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>#MainMenu,header,footer{visibility:hidden}.block-container{padding:0.5rem 0.5rem 0 0.5rem;max-width:100%}
div[data-testid="stButton"]{position:fixed;right:16px;bottom:16px;width:auto;z-index:99}div[data-testid="stButton"] button{width:auto;min-height:0;padding:6px 14px;border-radius:20px}</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models...")
def get_models():
    return service.load_models(MDIR, CITIES)


@st.cache_data(ttl=900, show_spinner="Downloading the latest data and forecasting...")
def build_payload(source):
    models = get_models()
    S = {"/api/refresh": {"ok": True}}
    note = ("Live hourly estimates from the Open-Meteo / CAMS air quality model, AQI computed with the CPCB method."
            if source == "live" else "Stored CPCB station data (ends Feb 2020), not live.")
    for c in CITIES:
        d = ds.load_city(source, c)
        S["/api/city/" + c] = service.city_payload(c, d)
        try:
            S["/api/forecast/" + c] = service.forecast_payload(c, d, models.get(c))
        except Exception as e:
            S["/api/forecast/" + c] = {"__err": "forecast failed: %s" % e}
        if not ds.STATUS.get(c, {}).get("live", True) and source == "live":
            note += " Download failed for %s, showing the last saved data." % c
    p = os.path.join(MDIR, "metrics.json")
    S["/api/metrics"] = json.load(open(p)) if os.path.exists(p) else {}
    S["/api/info"] = dict(source=source, note=note, cities=CITIES, status={})
    S["_built"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return S


if st.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

try:
    payload = build_payload(SOURCE)
except Exception as e:
    st.error("Could not load data: %s" % e)
    st.stop()

page = open(os.path.join(ROOT, "frontend", "index.html"), encoding="utf-8").read()
stub = ("const SAMPLE=" + json.dumps(payload, separators=(",", ":")).replace("</", "<\\/") + ";\n"
        "window.fetch=u=>{const p=decodeURIComponent(u.split('?')[0]),b=SAMPLE[p],e=b&&b.__err;"
        "return Promise.resolve({ok:b!==undefined&&!e,json:()=>Promise.resolve(e?{error:e}:b===undefined?{error:'no data'}:b)})};\n")
page = page.replace("<script>\n", "<script>\n" + stub, 1).replace("</style>", "#ref{display:none}.top{position:static}</style>", 1)
if hasattr(st, "iframe"):                       # newer Streamlit
    st.iframe(page, height=3700)
else:                                           # older Streamlit
    import streamlit.components.v1 as components
    components.html(page, height=3700, scrolling=True)
