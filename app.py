"""
RUL Predictor — NASA C-MAPSS FD004
Streamlit application for Remaining Useful Life prediction.

Models (all artifacts live in the './lstm/' folder):
  ✅ LSTM        — CNN-LSTM + Bahdanau Attention   (lstm_fd004_model.keras / lstm_fd004_preprocessors.pkl / lstm_fd004_config.pkl)
  ✅ Random Forest — Supervised Regression          (rf_best_model.pkl)
  ✅ XGBoost      — Supervised Regression           (xgb_best_model.pkl)
  ✅ K-Means      — Unsupervised Health Clustering  (kmeans_model.pkl / op_condition_kmeans.pkl / mm_scaler_cluster.pkl)

Extra pages:
  📊 EDA / Dataset Explorer  — live plots from the uploaded test_predictions.csv / train_regression.csv
  📈 Model Comparison        — head-to-head metric dashboard
  📖 About / Pipeline        — architecture & methodology writeup
"""

import io, json, os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings("ignore")

# ── Keras compatibility shim ─────────────────────────────────────────────────
try:
    import keras
    _register = keras.saving.register_keras_serializable
except (ImportError, AttributeError):
    from tensorflow import keras
    _register = tf.keras.utils.register_keras_serializable

# ── Colour palette ───────────────────────────────────────────────────────────
NASA_BLUE  = "#0B3D91"
NASA_RED   = "#FC3D21"
NASA_WHITE = "#FFFFFF"
NASA_LIGHT = "#E8EDF7"
NASA_GREY  = "#8A9BB5"

ASSET_DIR = "."   # all model files live next to app.py

# ── Page config — must be first Streamlit call ───────────────────────────────
st.set_page_config(
    page_title="NASA RUL Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "started" not in st.session_state:
    st.session_state.started = False

if st.query_params.get("start") == "true":
    st.session_state.started = True
    st.query_params.clear()
    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# INTRO SCREEN
# ════════════════════════════════════════════════════════════════════════════
intro_placeholder = st.empty()

if not st.session_state.started:
    with intro_placeholder.container():
        components.html("""
<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  html,body { width:100%; height:100%; overflow:hidden; background:#000814;
              font-family:'Courier New',Courier,monospace; }
  #stars { position:fixed; top:0; left:0; width:100%; height:100%; z-index:0; }
  .screen { position:relative; z-index:1; width:100%; height:100vh;
            display:flex; flex-direction:column; justify-content:center;
            align-items:center; gap:0; padding:20px; }
  .logo { width:160px; filter:drop-shadow(0 0 18px rgba(252,61,33,0.6));
          animation:float 4s ease-in-out infinite; margin-bottom:22px; }
  @keyframes float { 0%{transform:translateY(0px)} 50%{transform:translateY(-10px)} 100%{transform:translateY(0px)} }
  .label-ml { font-size:clamp(11px,1.8vw,15px); font-weight:700; letter-spacing:6px;
               color:#00B4D8; text-transform:uppercase; margin-bottom:10px;
               opacity:0; animation:fadeUp 0.8s ease 0.3s forwards; }
  .title-main { font-size:clamp(20px,4vw,38px); font-weight:900; letter-spacing:4px;
                color:#FFFFFF; text-transform:uppercase; text-align:center;
                text-shadow:0 0 30px rgba(0,180,216,0.4); margin-bottom:36px;
                opacity:0; animation:fadeUp 0.8s ease 0.6s forwards; }
  .divider { width:220px; height:1px;
             background:linear-gradient(90deg,transparent,#00B4D8,transparent);
             margin-bottom:28px; opacity:0; animation:fadeIn 0.8s ease 1s forwards; }
  .credits { display:flex; flex-direction:column; align-items:center; gap:8px;
             margin-bottom:40px; opacity:0; animation:fadeIn 0.8s ease 1.2s forwards; }
  .credit-name { font-size:clamp(12px,1.6vw,14px); letter-spacing:3px;
                  color:rgba(255,255,255,0.85); text-transform:uppercase; }
  .enter-btn { margin-top:6px; background:none;
               border:1px solid rgba(255,255,255,0.25); border-radius:30px;
               padding:10px 38px; color:rgba(255,255,255,0.9);
               font-family:'Courier New',Courier,monospace;
               font-size:clamp(10px,1.4vw,12px); letter-spacing:5px;
               text-transform:uppercase; cursor:pointer;
               opacity:0; animation:fadeIn 0.5s ease 2s forwards;
               transition:background 0.2s,border-color 0.2s; }
  .enter-btn:hover { background:rgba(0,180,216,0.15); border-color:#00B4D8; }
  .enter-btn span { animation:pulse 2.2s ease-in-out 2.5s infinite; display:inline-block; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
  @keyframes fadeIn { from{opacity:0} to{opacity:1} }
  @keyframes pulse { 0%,100%{opacity:0.35} 50%{opacity:0.08} }
</style></head><body>
<canvas id="stars"></canvas>
<div class="screen">
  <img class="logo" src="https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg" alt="NASA">
  <div class="label-ml">Machine Learning Project</div>
  <div class="title-main">Jet Engine RUL Prediction</div>
  <div class="divider"></div>
  <div class="credits">
    <div class="credit-name">Ali Ahsan</div>
    <div class="credit-name">Anas Bin Waheed</div>
    <div class="credit-name">Haqiq Azeem Khan</div>
  </div>
  <button class="enter-btn" onclick="goToApp()">
    <span>&#9654; &nbsp; Press Space or Click to Enter</span>
  </button>
</div>
<script>
const canvas=document.getElementById('stars'),ctx=canvas.getContext('2d');
function resize(){canvas.width=window.innerWidth;canvas.height=window.innerHeight;}
resize();window.addEventListener('resize',resize);
const stars=Array.from({length:180},()=>({x:Math.random(),y:Math.random(),r:Math.random()*1.4+0.3,a:Math.random(),da:(Math.random()*0.004+0.001)*(Math.random()<0.5?1:-1)}));
function drawStars(){ctx.clearRect(0,0,canvas.width,canvas.height);for(const s of stars){s.a=Math.max(0.05,Math.min(1,s.a+s.da));if(s.a<=0.05||s.a>=1)s.da*=-1;ctx.beginPath();ctx.arc(s.x*canvas.width,s.y*canvas.height,s.r,0,Math.PI*2);ctx.fillStyle=`rgba(255,255,255,${s.a})`;ctx.fill();}requestAnimationFrame(drawStars);}
drawStars();
function goToApp(){try{window.parent.location.href=window.parent.location.href.split('?')[0]+'?start=true';}catch(e){window.location.href='?start=true';}}
document.addEventListener('keydown',function(e){if(e.code==='Space'||e.key===' '){e.preventDefault();goToApp();}});
document.body.setAttribute('tabindex','0');document.body.focus();
</script></body></html>""", height=620, scrolling=False)
    st.stop()

intro_placeholder.empty()

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  html,body,[class*="css"]{{font-family:'Segoe UI','Helvetica Neue',Arial,sans-serif;}}
  .stApp{{background:linear-gradient(160deg,#010b1f 0%,#031638 60%,#0a1f4a 100%);color:{NASA_WHITE};}}
  [data-testid="stSidebar"]{{background:linear-gradient(180deg,#010d25 0%,#021238 100%);border-right:2px solid {NASA_BLUE};}}
  [data-testid="stSidebar"] .stMarkdown,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{{color:{NASA_WHITE}!important;}}
  h1{{color:{NASA_WHITE};letter-spacing:2px;}} h2,h3{{color:{NASA_LIGHT};}}
  .metric-card{{background:linear-gradient(135deg,#0a1a3a 0%,#0f2460 100%);border:1px solid {NASA_BLUE};border-radius:12px;padding:22px 26px;text-align:center;box-shadow:0 0 20px rgba(11,61,145,0.4);}}
  .metric-label{{font-size:13px;color:{NASA_GREY};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;}}
  .metric-value{{font-size:42px;font-weight:700;color:{NASA_WHITE};line-height:1.1;}}
  .metric-unit{{font-size:14px;color:{NASA_GREY};margin-top:4px;}}
  .rul-critical{{color:{NASA_RED};}} .rul-warning{{color:#FFA500;}} .rul-healthy{{color:#00C853;}}
  .status-badge{{display:inline-block;padding:5px 18px;border-radius:20px;font-size:13px;font-weight:600;letter-spacing:1px;text-transform:uppercase;}}
  .badge-critical{{background:rgba(252,61,33,0.2);border:1px solid {NASA_RED};color:{NASA_RED};}}
  .badge-warning{{background:rgba(255,165,0,0.2);border:1px solid #FFA500;color:#FFA500;}}
  .badge-healthy{{background:rgba(0,200,83,0.2);border:1px solid #00C853;color:#00C853;}}
  .info-box{{background:rgba(11,61,145,0.25);border-left:4px solid {NASA_BLUE};border-radius:6px;padding:14px 18px;margin:10px 0;font-size:14px;color:{NASA_LIGHT};}}
  .error-box{{background:rgba(252,61,33,0.15);border-left:4px solid {NASA_RED};border-radius:6px;padding:14px 18px;margin:10px 0;font-size:14px;color:#FFCDD2;}}
  .success-box{{background:rgba(0,200,83,0.12);border-left:4px solid #00C853;border-radius:6px;padding:14px 18px;margin:10px 0;font-size:14px;color:#C8E6C9;}}
  .template-box{{background:rgba(0,180,216,0.10);border:1px solid rgba(0,180,216,0.4);border-radius:8px;padding:16px 20px;margin:10px 0;font-size:13px;color:{NASA_LIGHT};}}
  .stTabs [data-baseweb="tab-list"]{{background:rgba(11,61,145,0.2);border-radius:10px;padding:4px;gap:4px;}}
  .stTabs [data-baseweb="tab"]{{border-radius:8px;color:{NASA_GREY};font-weight:600;font-size:14px;padding:8px 20px;}}
  .stTabs [aria-selected="true"]{{background:{NASA_BLUE}!important;color:{NASA_WHITE}!important;}}
  .stButton>button{{background:linear-gradient(135deg,{NASA_BLUE} 0%,#1a52c4 100%);color:{NASA_WHITE};border:none;border-radius:8px;font-weight:600;letter-spacing:0.5px;padding:10px 28px;transition:all 0.2s ease;}}
  .stButton>button:hover{{background:linear-gradient(135deg,#1a52c4 0%,#2563de 100%);box-shadow:0 4px 15px rgba(11,61,145,0.5);transform:translateY(-1px);}}
  [data-testid="stFileUploader"]{{background:rgba(11,61,145,0.15);border:2px dashed {NASA_BLUE};border-radius:10px;padding:10px;}}
  .stDataFrame{{border:1px solid {NASA_BLUE};border-radius:8px;}}
  hr{{border-color:rgba(11,61,145,0.4);}}
  .stSelectbox div[data-baseweb="select"]>div{{background:rgba(11,61,145,0.2);border-color:{NASA_BLUE};color:{NASA_WHITE};}}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# CUSTOM KERAS OBJECTS  (must be registered before load_model)
# ════════════════════════════════════════════════════════════════════════════
HUBER_DELTA       = 15.0
FAILURE_THRESHOLD = 50.0
FAILURE_WEIGHT    = 3.0

@_register(package="rul_predictor")
def weighted_huber_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    error  = y_true - y_pred
    abs_e  = tf.abs(error)
    huber  = tf.where(abs_e <= HUBER_DELTA,
                      0.5 * tf.square(error),
                      HUBER_DELTA * abs_e - 0.5 * HUBER_DELTA ** 2)
    weight = tf.where(y_true < FAILURE_THRESHOLD,
                      tf.ones_like(y_true) * FAILURE_WEIGHT,
                      tf.ones_like(y_true))
    return tf.reduce_mean(weight * huber)

@_register(package="rul_predictor")
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = tf.keras.layers.Dense(units, use_bias=True)
        self.v = tf.keras.layers.Dense(1, use_bias=False)
    def call(self, lstm_out):
        score   = self.v(tf.nn.tanh(self.W(lstm_out)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * lstm_out, axis=1)
        return context, weights
    def get_config(self):
        cfg = super().get_config(); cfg.update({"units": self.units}); return cfg

# ════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ════════════════════════════════════════════════════════════════════════════
def p(name):
    """Return full path inside ASSET_DIR."""
    return os.path.join(ASSET_DIR, name)

def rul_status(rul: float):
    if rul < 30:   return "CRITICAL", "badge-critical", "rul-critical", NASA_RED
    elif rul < 70: return "WARNING",  "badge-warning",  "rul-warning",  "#FFA500"
    else:          return "HEALTHY",  "badge-healthy",  "rul-healthy",  "#00C853"

def nasa_score(y_true, y_pred):
    diff = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1)))

def make_gauge(rul: float, rul_max: float = 125):
    _, _, _, colour = rul_status(rul)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=rul,
        number={"suffix": " cycles", "font": {"size": 32, "color": NASA_WHITE}},
        gauge={"axis": {"range": [0, rul_max], "tickcolor": NASA_GREY,
                        "tickfont": {"color": NASA_GREY, "size": 11}},
               "bar": {"color": colour, "thickness": 0.28},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 30],      "color": "rgba(252,61,33,0.18)"},
                         {"range": [30, 70],     "color": "rgba(255,165,0,0.13)"},
                         {"range": [70, rul_max],"color": "rgba(0,200,83,0.10)"}],
               "threshold": {"line": {"color": NASA_RED, "width": 2}, "thickness": 0.8, "value": 30}},
        title={"text": "Remaining Useful Life", "font": {"color": NASA_GREY, "size": 14}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color=NASA_WHITE, margin=dict(t=40,b=10,l=20,r=20), height=280)
    return fig

def render_prediction_result(rul_pred, rul_max, seq_raw=None, model_label=""):
    """Shared prediction result renderer (gauge + metrics + status)."""
    status, badge_cls, rul_cls, colour = rul_status(rul_pred)
    pct = (rul_pred / rul_max) * 100

    st.markdown("---")
    st.markdown(f"## 🎯 Prediction Result — {model_label}")

    col_gauge, col_metrics = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(make_gauge(rul_pred, rul_max), use_container_width=True)
    with col_metrics:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Predicted RUL</div>
          <div class="metric-value {rul_cls}">{rul_pred:.1f}</div>
          <div class="metric-unit">engine cycles</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Life Remaining</div>
              <div class="metric-value" style="font-size:30px;color:{colour};">{pct:.1f}%</div>
              <div class="metric-unit">of {rul_max}-cycle cap</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Status</div>
              <div style="margin-top:10px;"><span class="status-badge {badge_cls}">{status}</span></div>
              <div class="metric-unit" style="margin-top:10px;">
                {'⛔ Replace soon' if status=='CRITICAL' else '⚠️ Monitor closely' if status=='WARNING' else '✅ Normal operation'}
              </div>
            </div>""", unsafe_allow_html=True)

    interp = {
        "CRITICAL": f"**⛔ CRITICAL — Immediate action required.** Predicted RUL of **{rul_pred:.1f} cycles** is below the 30-cycle threshold. Schedule maintenance or replacement immediately.",
        "WARNING":  f"**⚠️ WARNING — Schedule maintenance.** Predicted RUL of **{rul_pred:.1f} cycles** places this engine in the failure zone (< 70 cycles). Plan maintenance soon.",
        "HEALTHY":  f"**✅ HEALTHY — Normal operation.** Predicted RUL of **{rul_pred:.1f} cycles** indicates good engine health. Continue routine monitoring.",
    }
    st.info(interp[status])

    if seq_raw is not None:
        st.markdown("---")
        st.markdown("#### 🌡️ Input Sequence Heatmap (last 30 cycles × features)")
        n_feat = seq_raw.shape[1]
        # Build accurate axis labels using actual sensor names and the stored op_condition column index.
        # The heatmap renders seq_raw which came from num_df.iloc[:, :N_FEAT] in column order.
        # For the template / scenario CSVs that order is LSTM_SENSORS + [op_condition] (op_condition last).
        # We use op_feat_idx from preprocessors to place the label correctly.
        # Label columns using known LSTM feature order: LSTM_SENSORS followed by op_condition.
        # seq_raw is sliced from num_df.iloc[:, :N_FEAT] whose numeric columns follow
        # the CSV column order — the template/scenario CSVs always put sensors first then op_condition.
        base_labels = list(LSTM_SENSORS) + ["op_condition"]
        feat_names = base_labels[:n_feat]
        fig = go.Figure(go.Heatmap(
            z=seq_raw.T,
            x=[f"t-{30-i}" for i in range(30)],
            y=feat_names, colorscale="RdBu_r", showscale=True,
            colorbar={"tickfont": {"color": NASA_GREY}},
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color=NASA_WHITE, margin=dict(t=10,b=40,l=90,r=20), height=340,
                          xaxis={"tickfont":{"color":NASA_GREY},"title":{"text":"Timestep","font":{"color":NASA_GREY}}},
                          yaxis={"tickfont":{"color":NASA_GREY,"size":10}})
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# RESOURCE LOADERS  (cached)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_lstm_artifacts():
    model = tf.keras.models.load_model(
        p("lstm_fd004_model.keras"),
        custom_objects={"weighted_huber_loss": weighted_huber_loss,
                        "BahdanauAttention": BahdanauAttention},
    )
    pre = joblib.load(p("lstm_fd004_preprocessors.pkl"))
    cfg = joblib.load(p("lstm_fd004_config.pkl"))
    return model, pre, cfg

@st.cache_resource(show_spinner=False)
def load_rf():
    return joblib.load(p("rf_best_model.pkl"))

@st.cache_resource(show_spinner=False)
def load_xgb():
    return joblib.load(p("xgb_best_model.pkl"))

@st.cache_resource(show_spinner=False)
def load_kmeans_artifacts():
    km    = joblib.load(p("kmeans_model.pkl"))
    km_op = joblib.load(p("op_condition_kmeans.pkl"))
    mm    = joblib.load(p("mm_scaler_cluster.pkl"))
    return km, km_op, mm

@st.cache_data(show_spinner=False)
def load_test_predictions():
    fp = p("test_predictions.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp)
    return None

@st.cache_data(show_spinner=False)
def load_train_data():
    fp = p("train_regression.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp)
    return None

# ════════════════════════════════════════════════════════════════════════════
# PREPROCESSING HELPERS
# ════════════════════════════════════════════════════════════════════════════
SENSOR_COLS_ALL = [f"sensor_{i:02d}" for i in range(1, 22)]

# Surviving sensors — exactly the 12 that remain in train_regression.csv after
# dropping constant ones (sensor_05,06,10,11,15,16,19,20,21 removed) plus op_set_3.
# These are the ONLY sensor columns the RF/XGBoost models were trained on.
SURVIVING_SENSORS = [
    "sensor_01", "sensor_02", "sensor_03", "sensor_04",
    "sensor_07", "sensor_08", "sensor_09", "sensor_12",
    "sensor_13", "sensor_14", "sensor_17", "sensor_18"
]

def add_rolling_features(df, sensor_cols, window=30):
    df = df.sort_values(["unit_id", "cycle"]).copy()
    new = {}
    for col in sensor_cols:
        g = df.groupby("unit_id")[col]
        new[f"{col}_rmean"] = g.transform(lambda x: x.rolling(window, min_periods=1).mean())
        new[f"{col}_rstd"]  = g.transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

def prepare_rf_xgb_row(df_engine, model):
    """
    Given a DataFrame of one engine's sensor data (from CSV upload),
    compute rolling stats and return the feature vector for the LAST row.
    Mirrors exactly what Part 2/3 of the notebook does:
      - detect sensor cols (sensor_XX names present in df)
      - add 30-cycle rolling mean + std per sensor
      - drop unit_id, cycle, RUL, RUL_clipped, op_set_1/2/3, max_cycle
      - keep everything else (sensors + op_condition + rolling features)
    """
    sensor_cols = [c for c in SURVIVING_SENSORS if c in df_engine.columns]
    df_engine = df_engine.sort_values("cycle").copy()
    df_engine = add_rolling_features(df_engine, sensor_cols, window=30)

    drop_like = ["unit_id", "cycle", "RUL", "RUL_clipped",
                 "op_set_1", "op_set_2", "op_set_3", "max_cycle"]
    feat_cols = [c for c in df_engine.columns if c not in drop_like]

    # Use model's expected features if available (most reliable)
    try:
        expected = model.feature_names_in_
        feat_cols = [c for c in expected if c in df_engine.columns]
    except AttributeError:
        pass

    row = df_engine[feat_cols].iloc[[-1]].fillna(0)
    return row, feat_cols

# ════════════════════════════════════════════════════════════════════════════
# SCENARIO TEMPLATES  (3 health states × 3 models)
# ════════════════════════════════════════════════════════════════════════════

LSTM_SENSORS = ["sensor_02","sensor_03","sensor_04","sensor_07","sensor_08",
                "sensor_09","sensor_11","sensor_12","sensor_13","sensor_14",
                "sensor_15","sensor_17"]

# ── Real-data calibrated drift & start values ─────────────────────────────
# Derived by analysing 249 training engines across three RUL zones:
#   healthy  = first 45 rows per engine  (RUL_clipped == 125)
#   moderate = 45-row window centred at  RUL_clipped ≈ 60
#   critical = last  45 rows per engine  (RUL_clipped ≤ 20)
# Values are StandardScaler-normalised — same scale the models were trained on.
# sensor_11 / sensor_15 are LSTM-only; proxied from sensor_09 / sensor_03 respectively.

_REAL_DRIFT = {
    "healthy":  {"sensor_01": 0.153,"sensor_02": 0.122,"sensor_03": 0.095,"sensor_04": 0.097,
                 "sensor_07": 0.167,"sensor_08":-0.009,"sensor_09": 0.081,"sensor_12": 0.166,
                 "sensor_13":-0.099,"sensor_14":-0.057,"sensor_17": 0.095,"sensor_18":-0.009,
                 "sensor_11": 0.081,"sensor_15": 0.095},
    "moderate": {"sensor_01": 0.011,"sensor_02": 0.002,"sensor_03": 0.070,"sensor_04": 0.116,
                 "sensor_07": 0.019,"sensor_08":-0.058,"sensor_09": 0.062,"sensor_12": 0.020,
                 "sensor_13":-0.075,"sensor_14": 0.206,"sensor_17": 0.074,"sensor_18":-0.060,
                 "sensor_11": 0.062,"sensor_15": 0.070},
    "critical": {"sensor_01": 0.283,"sensor_02": 0.223,"sensor_03": 0.330,"sensor_04": 0.416,
                 "sensor_07": 0.260,"sensor_08":-0.038,"sensor_09": 0.295,"sensor_12": 0.261,
                 "sensor_13":-0.194,"sensor_14": 0.408,"sensor_17": 0.323,"sensor_18":-0.043,
                 "sensor_11": 0.295,"sensor_15": 0.330},
}

# Target rolling means extracted directly from real FD004 training windows:
#   healthy  = mean of 30-cycle rolling means across first-45-row windows (RUL_clipped=125)
#   moderate = mean across 45-row windows centred at RUL_clipped ≈ 60
#   critical = mean across last-45-row windows (RUL_clipped ≤ 20)
_TARGET_RMEAN = {
    "healthy":  {"sensor_01":-0.006,"sensor_02":-0.016,"sensor_03":-0.075,"sensor_04":-0.101,
                 "sensor_07":-0.013,"sensor_08": 0.007,"sensor_09":-0.058,"sensor_12":-0.013,
                 "sensor_13": 0.012,"sensor_14":-0.153,"sensor_17":-0.075,"sensor_18": 0.008,
                 "sensor_11":-0.058,"sensor_15":-0.075},
    "moderate": {"sensor_01": 0.018,"sensor_02": 0.018,"sensor_03": 0.031,"sensor_04": 0.039,
                 "sensor_07": 0.016,"sensor_08": 0.000,"sensor_09": 0.020,"sensor_12": 0.016,
                 "sensor_13":-0.009,"sensor_14": 0.021,"sensor_17": 0.032,"sensor_18": 0.000,
                 "sensor_11": 0.020,"sensor_15": 0.031},
    "critical": {"sensor_01":-0.003,"sensor_02": 0.035,"sensor_03": 0.168,"sensor_04": 0.223,
                 "sensor_07": 0.019,"sensor_08": 0.020,"sensor_09": 0.143,"sensor_12": 0.019,
                 "sensor_13": 0.024,"sensor_14": 0.396,"sensor_17": 0.167,"sensor_18": 0.016,
                 "sensor_11": 0.143,"sensor_15": 0.168},
}

_RF_SEED_OFF          = 100  # healthy/moderate RF get different noise pattern than critical
_XGB_SEED_OFF         = 500
_CRITICAL_SHARED_SEED = 0   # RF and XGB share seed on critical → identical CSV (any RUL diff = model diff only)

SCENARIO_META = {
    "healthy": {
        "label":  "✅ Healthy Engine",
        "emoji":  "✅",
        "color":  "#00C853",
        "badge":  "badge-healthy",
        "expected_rul": "~90–125 cycles",
        "explanation": (
            "Built from real FD004 training data: first-45-row windows of 249 engines "
            "with RUL_clipped = 125 (early life). Per-sensor target rolling means are "
            "extracted from those real windows — <b>sensor_14 rmean ≈ −0.153</b> "
            "(cold HPC, no blade clearance opening). "
            "Sensor values are constructed so the 30-cycle rolling mean lands exactly on "
            "the real training mean, and rolling std ≈ 0.94 matches unit-variance "
            "StandardScaler data. Expected: <b>RUL near the 125-cycle cap</b>."
        ),
    },
    "moderate": {
        "label":  "⚠️ Moderate Degradation",
        "emoji":  "⚠️",
        "color":  "#FFA500",
        "badge":  "badge-warning",
        "expected_rul": "~40–80 cycles",
        "explanation": (
            "Built from 45-row windows centred at RUL_clipped ≈ 60 across all 249 "
            "training engines. <b>sensor_14 rmean shifts to +0.021</b> — rising above zero "
            "marks the onset of measurable HPC degradation. sensor_14 drift = +0.21 over "
            "45 cycles. Rolling mean of sensor_14 at last row ≈ +0.056 (real target: +0.021). "
            "Both RF and XGBoost will detect the elevated mean and output a mid-range RUL. "
            "Expected: <b>40–80 cycles remaining</b>."
        ),
    },
    "critical": {
        "label":  "⛔ Critical / Near Failure",
        "emoji":  "⛔",
        "color":  "#FC3D21",
        "badge":  "badge-critical",
        "expected_rul": "~0–30 cycles",
        "explanation": (
            "Built from last-45-row windows of real engines with RUL_clipped ≤ 20. "
            "<b>RF and XGBoost receive the identical CSV</b> (shared seed) — any RUL "
            "difference reflects genuine model boundary differences, not data noise. "
            "<b>sensor_14 rmean = +0.396</b>, sensor_04 = +0.223, sensor_03 = +0.168. "
            "sensor_13 drops (fan degradation marker). Rolling std ≈ 0.94. "
            "Expected: <b>0–30 cycles — immediate maintenance required</b>."
        ),
    },
}


def _make_sensor_window(target_rmean: float, drift: float,
                        n: int, window: int = 30, seed: int = 42) -> np.ndarray:
    """
    Build n sensor readings with a controlled rolling mean over the last `window` rows.
    Strategy:
      - Linear trend centred on target_rmean with total change = drift
      - Gaussian noise std=0.9 (matching real StandardScaler unit variance → rolling std ~0.94)
      - Last `window` rows of noise are zero-mean by construction, so the rolling mean
        at the last row equals the mean of the trend over those rows.
      - When n == window the rolling mean equals target_rmean exactly.
        When n > window (e.g. n=45, window=30) the rolling mean over the last 30 rows
        equals target_rmean + drift*(n - window - 1) / (2*(n - 1)) approximately,
        which is slightly above target_rmean when drift > 0 (right half of the trend).
    This still produces values well within the correct RUL zone for each scenario.
    """
    np.random.seed(seed)
    trend = np.linspace(target_rmean - drift / 2, target_rmean + drift / 2, n)
    noise = np.random.normal(0, 0.9, n)
    noise[-window:] -= noise[-window:].mean()   # force last window noise to zero-mean
    return np.round(trend + noise, 4)


def make_lstm_scenario(scenario: str, n: int = 35) -> pd.DataFrame:
    """35-row LSTM CSV with guaranteed correct rolling mean per RUL zone."""
    data = {}
    for i, s in enumerate(LSTM_SENSORS):
        data[s] = _make_sensor_window(
            _TARGET_RMEAN[scenario][s], _REAL_DRIFT[scenario][s],
            n=n, window=min(30, n), seed=42 + i * 7 + 1000)
    data["op_condition"] = [i % 6 for i in range(n)]
    return pd.DataFrame(data)


def make_rf_scenario(scenario: str, n: int = 45) -> pd.DataFrame:
    """45-row RF CSV with guaranteed correct rolling mean per RUL zone."""
    seed_off = _CRITICAL_SHARED_SEED if scenario == "critical" else _RF_SEED_OFF
    data = {"unit_id": [1] * n, "cycle": list(range(1, n + 1))}
    for i, s in enumerate(SURVIVING_SENSORS):
        data[s] = _make_sensor_window(
            _TARGET_RMEAN[scenario][s], _REAL_DRIFT[scenario][s],
            n=n, window=30, seed=42 + i * 7 + seed_off)
    data["op_condition"] = [i % 6 for i in range(n)]
    return pd.DataFrame(data)


def make_xgb_scenario(scenario: str, n: int = 45) -> pd.DataFrame:
    """45-row XGBoost CSV. Critical: same seed as RF (identical data = pure model comparison)."""
    seed_off = _CRITICAL_SHARED_SEED if scenario == "critical" else _XGB_SEED_OFF
    data = {"unit_id": [1] * n, "cycle": list(range(1, n + 1))}
    for i, s in enumerate(SURVIVING_SENSORS):
        data[s] = _make_sensor_window(
            _TARGET_RMEAN[scenario][s], _REAL_DRIFT[scenario][s],
            n=n, window=30, seed=42 + i * 7 + seed_off)
    data["op_condition"] = [i % 6 for i in range(n)]
    return pd.DataFrame(data)


def make_lstm_template(n_rows=35):
    return make_lstm_scenario("moderate", n=n_rows)

def make_rf_xgb_template(n_rows=45):
    return make_rf_scenario("moderate", n=n_rows)


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ════════════════════════════════════════════════════════════════════════════
# SHARED SCENARIO RUNNER WIDGET
# Used by LSTM, RF, and XGBoost tabs to display the 3-scenario panel
# ════════════════════════════════════════════════════════════════════════════
def render_scenario_panel(model_type: str, model_ok: bool,
                          run_fn,        # callable(df) -> float RUL
                          rul_max: int = 125):
    """
    Renders the 3-scenario 'Try Sample Engine' panel.
    model_type: 'lstm' | 'rf' | 'xgb'
    run_fn: function that accepts a DataFrame and returns predicted RUL (float)
    """
    st.markdown("---")
    st.markdown("### 🧪 Try a Sample Engine — No Upload Needed")
    st.markdown(f"""
    <div class="info-box" style="font-size:13px;">
      Click any scenario below to instantly run the model on a synthetic engine with
      known health characteristics. Use this to verify the model is working correctly
      and to understand what different degradation patterns look like in practice.
      Download buttons are also available for each scenario CSV.
    </div>""", unsafe_allow_html=True)

    cols = st.columns(3)
    scenarios = ["healthy", "moderate", "critical"]

    for col, sc in zip(cols, scenarios):
        meta = SCENARIO_META[sc]
        with col:
            st.markdown(f"""
            <div style="border:1px solid {meta['color']};border-radius:10px;
                        padding:14px 16px;background:rgba(0,0,0,0.2);margin-bottom:8px;">
              <div style="font-size:16px;font-weight:700;color:{meta['color']};
                          margin-bottom:6px;">{meta['label']}</div>
              <div style="font-size:12px;color:{NASA_LIGHT};line-height:1.5;">
                Expected RUL: <b style="color:{meta['color']}">{meta['expected_rul']}</b>
              </div>
            </div>""", unsafe_allow_html=True)

            # explanation expander
            with st.expander("💡 Why this RUL?", expanded=False):
                st.markdown(f"<div style='font-size:12px;color:{NASA_LIGHT};'>{meta['explanation']}</div>",
                            unsafe_allow_html=True)

            # build the right df
            if model_type == "lstm":
                df_sc = make_lstm_scenario(sc)
            elif model_type == "rf":
                df_sc = make_rf_scenario(sc)
            else:
                df_sc = make_xgb_scenario(sc)

            btn_key  = f"try_{model_type}_{sc}"
            dl_key   = f"dl_{model_type}_{sc}"

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                run_it = st.button(f"{meta['emoji']} Run", key=btn_key,
                                   use_container_width=True,
                                   disabled=not model_ok)
            with bcol2:
                st.download_button("⬇️ CSV", data=df_to_csv_bytes(df_sc),
                                   file_name=f"{model_type}_{sc}.csv",
                                   mime="text/csv", key=dl_key,
                                   use_container_width=True)

            if run_it:
                if not model_ok:
                    st.markdown('<div class="error-box">Model not loaded.</div>',
                                unsafe_allow_html=True)
                else:
                    with st.spinner(f"🧠 Running {model_type.upper()} on {sc} engine…"):
                        try:
                            rul_pred = run_fn(df_sc)
                            status, badge_cls, rul_cls, colour = rul_status(rul_pred)
                            pct = (rul_pred / rul_max) * 100
                            st.markdown(f"""
                            <div style="border:2px solid {colour};border-radius:10px;
                                        padding:14px;margin-top:8px;text-align:center;">
                              <div style="font-size:11px;color:{NASA_GREY};
                                          text-transform:uppercase;letter-spacing:1px;">
                                Predicted RUL</div>
                              <div style="font-size:36px;font-weight:800;
                                          color:{colour};line-height:1.1;">
                                {rul_pred:.1f}</div>
                              <div style="font-size:12px;color:{NASA_GREY};">cycles</div>
                              <div style="margin-top:8px;">
                                <span class="status-badge {badge_cls}">{status}</span>
                              </div>
                              <div style="font-size:11px;color:{NASA_GREY};margin-top:6px;">
                                {pct:.1f}% life remaining
                              </div>
                            </div>""", unsafe_allow_html=True)
                        except Exception as exc:
                            st.markdown(f'<div class="error-box">❌ {exc}</div>',
                                        unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# LSTM PREDICT
# ════════════════════════════════════════════════════════════════════════════
def predict_rul_lstm(model, seq, preprocessors, config):
    seq_len = config["SEQ_LEN"]; n_feat = config["N_FEAT"]; rul_max = config["RUL_MAX"]
    op_idx = preprocessors["op_feat_idx"]; op_min = preprocessors["op_min"]; op_max = preprocessors["op_max"]
    if seq.shape != (seq_len, n_feat):
        raise ValueError(f"Expected ({seq_len}, {n_feat}), got {seq.shape}")
    seq = seq.astype(np.float32).copy()
    seq[:, op_idx] = 2.0 * (seq[:, op_idx] - op_min) / (op_max - op_min) - 1.0
    raw = model.predict(seq.reshape(1, seq_len, n_feat), verbose=0).flatten()[0]
    return float(np.clip(raw, 0.0, rul_max))

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:10px 0 20px 0;">
      <div style="font-size:40px;">🚀</div>
      <div style="font-size:20px;font-weight:700;color:{NASA_WHITE};letter-spacing:2px;">RUL PREDICTOR</div>
      <div style="font-size:11px;color:{NASA_GREY};letter-spacing:3px;margin-top:4px;">NASA C-MAPSS · FD004</div>
    </div>
    <hr style="border-color:rgba(11,61,145,0.5);margin:0 0 20px 0;">
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="font-size:12px;">
      <b>📂 Artifact folder:</b><br>
      <code>./lstm/</code><br><br>
      All model files should be placed inside this folder.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**ℹ️ Model Results**")
    st.markdown(f"""
    <div class="info-box" style="font-size:12px;">
      <b>LSTM:</b> RMSE 22.7 · NASA 3994 (237 engines)<br>
      <b>Random Forest:</b> RMSE ~27.6 · NASA ~34084<br>
      <b>XGBoost:</b> RMSE ~25.1 · NASA ~8326<br>
      <b>K-Means:</b> Silhouette 0.43 · DB 0.85
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:{NASA_GREY};text-align:center;'>NASA C-MAPSS Prognostics · v2.0</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HEADER BANNER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:linear-gradient(90deg,{NASA_BLUE} 0%,#1040a0 50%,{NASA_BLUE} 100%);
  border-bottom:3px solid {NASA_RED};padding:22px 30px 18px 30px;border-radius:10px;
  margin-bottom:28px;display:flex;align-items:center;gap:20px;">
  <div style="font-size:52px;line-height:1;">🛰️</div>
  <div>
    <div style="font-size:26px;font-weight:800;letter-spacing:3px;color:{NASA_WHITE};">ENGINE HEALTH ANALYTICS</div>
    <div style="font-size:13px;color:rgba(255,255,255,0.7);letter-spacing:2px;margin-top:4px;">
      REMAINING USEFUL LIFE PREDICTION · NASA C-MAPSS FD004
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab_lstm, tab_rf, tab_xgb, tab_km, tab_eda, tab_compare, tab_about = st.tabs([
    "⚡ LSTM",
    "🌲 Random Forest",
    "🔥 XGBoost",
    "🔵 K-Means Explorer",
    "📊 EDA",
    "📈 Model Comparison",
    "📖 About",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — LSTM
# ════════════════════════════════════════════════════════════════════════════
with tab_lstm:
    st.markdown("### ⚡ LSTM — CNN + LSTM + Bahdanau Attention")
    st.markdown(f"""
    <div class="info-box">
      Upload a CSV with engine sensor time-series data.
      The app extracts the <b>last 30 timesteps × 13 features</b> and predicts RUL.<br><br>
      <b>Required columns:</b> 13 numeric feature columns
      (12 per-condition StandardScaled sensors + op_condition as integer 0–5).
      The file may have any number of rows ≥ 30 and optional metadata columns.
    </div>
    """, unsafe_allow_html=True)

    # ── Template download ────────────────────────────────────────────────────
    with st.expander("📥 Download CSV Template for LSTM", expanded=False):
        st.markdown(f"""
        <div class="template-box">
          <b>What this template contains:</b><br>
          • <b>35 rows</b> (minimum 30 needed) of simulated sensor data<br>
          • <b>12 sensor columns</b>: <code>sensor_02, 03, 04, 07, 08, 09, 11, 12, 13, 14, 15, 17</code> — values are <b>StandardScaler-normalised</b> (roughly −3 to +3 range). Note: this is a non-contiguous set; sensors 01, 05, 06, 10, 16, 18 are absent from the LSTM feature set.<br>
          • <b>op_condition</b> column — integer 0–5 representing operating condition cluster<br><br>
          <b>How to use:</b> Download, open in Excel or any editor, replace the values with your
          actual sensor data (already normalised per operating condition), then upload below.
          The app will use the last 30 rows automatically.
        </div>
        """, unsafe_allow_html=True)
        lstm_template = make_lstm_template(35)
        st.dataframe(lstm_template.head(5), use_container_width=True)
        st.caption("Preview of first 5 rows. Template has 35 rows total.")
        st.download_button(
            label="⬇️ Download LSTM Template CSV",
            data=df_to_csv_bytes(lstm_template),
            file_name="lstm_template.csv",
            mime="text/csv",
            key="lstm_template_dl"
        )

    model_ok = False
    try:
        with st.spinner("🔄 Loading LSTM model artifacts…"):
            lstm_model, lstm_pre, lstm_cfg = load_lstm_artifacts()
        model_ok = True
        st.markdown('<div class="success-box">✅ Model loaded — ready for inference.</div>', unsafe_allow_html=True)
    except FileNotFoundError as e:
        st.markdown(f'<div class="error-box">⚠️ File not found: <code>{e}</code><br>Place artifacts in <code>./lstm/</code>.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ Error loading model: {e}</div>', unsafe_allow_html=True)

    # ── Sample-engine scenario panel ─────────────────────────────────────────
    def _run_lstm(df_sc):
        SEQ_LEN = lstm_cfg["SEQ_LEN"]; N_FEAT = lstm_cfg["N_FEAT"]
        num_df = df_sc.select_dtypes(include=[np.number])
        if num_df.shape[1] < N_FEAT or len(num_df) < SEQ_LEN:
            raise ValueError(f"Scenario df too small: {num_df.shape}")
        seq_raw = num_df.iloc[:, :N_FEAT].values[-SEQ_LEN:].astype(np.float32)
        return predict_rul_lstm(lstm_model, seq_raw, lstm_pre, lstm_cfg)

    render_scenario_panel("lstm", model_ok, _run_lstm, rul_max=lstm_cfg["RUL_MAX"] if model_ok else 125)

    st.markdown("---")
    st.markdown("### 📁 Or Upload Your Own CSV")
    col_up, col_hint = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("📁 Upload engine sensor CSV (LSTM)", type=["csv"], key="lstm_upload",
                                    help="CSV with ≥ 30 rows and 13 numeric feature columns.")
    with col_hint:
        st.markdown(f"""
        <div style="padding-top:8px;">
        <div class="info-box" style="font-size:12px;">
          <b>CSV format:</b><br>
          • ≥ 30 data rows<br>
          • 13 numeric feature cols<br>
          • 12 StandardScaled sensors + op_condition (0–5)<br>
          • Extra columns (metadata) are ignored<br>
          • Header row required
        </div></div>""", unsafe_allow_html=True)

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            st.markdown(f"**Preview** — {len(raw_df)} rows × {raw_df.shape[1]} columns")
            with st.expander("📊 Data preview (first 10 rows)", expanded=False):
                st.dataframe(raw_df.head(10), use_container_width=True)

            SEQ_LEN = lstm_cfg["SEQ_LEN"]; N_FEAT = lstm_cfg["N_FEAT"]
            num_df = raw_df.select_dtypes(include=[np.number])

            # Validation with helpful messages
            if num_df.shape[1] < N_FEAT:
                st.markdown(f'<div class="error-box">❌ Need at least <b>{N_FEAT} numeric columns</b>, found <b>{num_df.shape[1]}</b>.<br>'
                            f'Your CSV has {raw_df.shape[1]} total columns but only {num_df.shape[1]} are numeric. '
                            f'Download the template above to see the expected format.</div>', unsafe_allow_html=True)
            elif len(num_df) < SEQ_LEN:
                st.markdown(f'<div class="error-box">❌ Need at least <b>{SEQ_LEN} rows</b>, found <b>{len(num_df)}</b>.<br>'
                            f'The LSTM uses a sliding window of 30 timesteps. Add more rows to your CSV.</div>', unsafe_allow_html=True)
            else:
                feat_df = num_df.iloc[:, :N_FEAT]
                seq_raw = feat_df.values[-SEQ_LEN:].astype(np.float32)
                st.markdown(f'<div class="success-box">✅ Extracted sequence: <b>({SEQ_LEN}, {N_FEAT})</b> from last {SEQ_LEN} rows. '
                            f'Using columns: <code>{", ".join(feat_df.columns.tolist())}</code></div>', unsafe_allow_html=True)
                if model_ok:
                    with st.spinner("🧠 Running LSTM inference…"):
                        rul_pred = predict_rul_lstm(lstm_model, seq_raw, lstm_pre, lstm_cfg)
                    render_prediction_result(rul_pred, lstm_cfg["RUL_MAX"], seq_raw, "LSTM")
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error: {e}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:40px 20px;border:2px dashed rgba(11,61,145,0.4);border-radius:14px;margin-top:20px;">
          <div style="font-size:48px;margin-bottom:14px;">📡</div>
          <div style="font-size:17px;font-weight:600;color:{NASA_WHITE};margin-bottom:8px;">No file uploaded yet</div>
          <div style="font-size:13px;color:{NASA_GREY};max-width:480px;margin:0 auto;">
            Use the <b>Try a Sample Engine</b> panel above to run the model instantly,
            or upload your own CSV (≥ 30 rows, 13 numeric feature columns).
          </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RANDOM FOREST
# ════════════════════════════════════════════════════════════════════════════
with tab_rf:
    st.markdown("### 🌲 Random Forest — Supervised Regression")
    st.markdown(f"""
    <div class="info-box">
      Random Forest operates on <b>flat feature vectors</b>, not sequences.
      Upload a CSV of one engine's full history (≥ 30 rows).
      The app adds rolling statistics (window = 30 cycles) and predicts RUL from the last row.<br><br>
      <b>Required columns:</b> <code>unit_id</code>, <code>cycle</code>,
      the 12 surviving sensor columns (<code>sensor_01, 02, 03, 04, 07, 08, 09, 12, 13, 14, 17, 18</code>),
      and <code>op_condition</code> (integer 0–5).<br><br>
      <b>Sensor values</b> should be <b>StandardScaler-normalised</b> (roughly −3 to +3 range),
      matching the per-condition scaling applied during training.
    </div>
    """, unsafe_allow_html=True)

    # ── Template download ────────────────────────────────────────────────────
    with st.expander("📥 Download CSV Template for Random Forest", expanded=False):
        st.markdown(f"""
        <div class="template-box">
          <b>What this template contains:</b><br>
          • <b>45 rows</b> of simulated engine history (minimum 30 needed for rolling stats)<br>
          • <b>unit_id</b> — engine identifier (use 1 for a single engine)<br>
          • <b>cycle</b> — sequential cycle counter starting at 1<br>
          • <b>12 sensor columns</b> (<code>sensor_01, 02, 03, 04, 07, 08, 09, 12, 13, 14, 17, 18</code>) — StandardScaler-normalised values (~−3 to +3 range)<br>
          • <b>op_condition</b> — integer 0–5 (operating condition cluster label)<br><br>
          <b>Sensors NOT included</b> (constant/dropped during preprocessing: sensor_05, 06, 10, 11, 15, 16, 19, 20, 21).<br><br>
          <b>How to use:</b> Download, replace the sensor values with your actual normalised readings, upload below.
          The app computes 30-cycle rolling mean + std automatically and uses the last row for prediction.
        </div>
        """, unsafe_allow_html=True)
        rf_template = make_rf_xgb_template(45)
        st.dataframe(rf_template.head(5), use_container_width=True)
        st.caption("Preview of first 5 rows. Template has 45 rows total.")
        st.download_button(
            label="⬇️ Download Random Forest Template CSV",
            data=df_to_csv_bytes(rf_template),
            file_name="rf_template.csv",
            mime="text/csv",
            key="rf_template_dl"
        )

    rf_ok = False
    try:
        with st.spinner("🔄 Loading Random Forest…"):
            rf_model = load_rf()
        rf_ok = True
        st.markdown('<div class="success-box">✅ Random Forest loaded — ready for inference.</div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown(f'<div class="error-box">⚠️ <code>rf_best_model.pkl</code> not found in <code>./lstm/</code>.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ Error loading RF: {e}</div>', unsafe_allow_html=True)

    # ── Sample-engine scenario panel ─────────────────────────────────────────
    def _run_rf(df_sc):
        if "unit_id" not in df_sc.columns:
            df_sc = df_sc.copy(); df_sc.insert(0, "unit_id", 1)
        if "cycle" not in df_sc.columns:
            df_sc = df_sc.copy(); df_sc.insert(1, "cycle", range(1, len(df_sc)+1))
        X_row, _ = prepare_rf_xgb_row(df_sc, rf_model)
        return float(np.clip(rf_model.predict(X_row)[0], 0, 125))

    render_scenario_panel("rf", rf_ok, _run_rf)

    st.markdown("---")
    st.markdown("### 📁 Or Upload Your Own CSV")
    col_up2, col_hint2 = st.columns([2, 1])
    with col_up2:
        rf_upload = st.file_uploader("📁 Upload engine sensor CSV (Random Forest)", type=["csv"], key="rf_upload")
    with col_hint2:
        st.markdown(f"""
        <div style="padding-top:8px;">
        <div class="info-box" style="font-size:12px;">
          <b>Required columns:</b><br>
          • unit_id (engine ID)<br>
          • cycle (sequential int)<br>
          • sensor_01, 02, 03, 04, 07, 08, 09, 12, 13, 14, 17, 18<br>
          • op_condition (int 0–5)<br>
          • ≥ 30 rows needed<br>
          • Values: StandardScaled (~−3 to +3)<br>
          • No RUL column required
        </div></div>""", unsafe_allow_html=True)

    if rf_upload is not None:
        try:
            rf_df = pd.read_csv(rf_upload)
            st.markdown(f"**Preview** — {len(rf_df)} rows × {rf_df.shape[1]} columns")
            with st.expander("📊 Data preview (first 10 rows)", expanded=False):
                st.dataframe(rf_df.head(10), use_container_width=True)

            # Auto-add missing structural columns
            if "unit_id" not in rf_df.columns:
                rf_df.insert(0, "unit_id", 1)
                st.markdown('<div class="info-box" style="font-size:12px;">ℹ️ <code>unit_id</code> column not found — defaulted to 1.</div>', unsafe_allow_html=True)
            if "cycle" not in rf_df.columns:
                rf_df.insert(1, "cycle", range(1, len(rf_df)+1))
                st.markdown('<div class="info-box" style="font-size:12px;">ℹ️ <code>cycle</code> column not found — auto-generated as 1…N.</div>', unsafe_allow_html=True)

            # Warn if multiple engines detected — only the last engine in the file is used
            n_engines = rf_df["unit_id"].nunique()
            if n_engines > 1:
                last_engine_id = rf_df["unit_id"].iloc[-1]
                st.markdown(f'<div class="info-box" style="font-size:12px;">⚠️ <b>Multiple engines detected ({n_engines} units).</b> '
                            f'This tab predicts for <b>one engine at a time</b>. Only the last engine '
                            f'(<code>unit_id={last_engine_id}</code>) will be used for prediction.<br>'
                            f'To predict a specific engine, upload a CSV filtered to that engine only.</div>', unsafe_allow_html=True)
                rf_df = rf_df[rf_df["unit_id"] == last_engine_id].copy()

            # Check for sensor columns
            found_sensors = [c for c in SURVIVING_SENSORS if c in rf_df.columns]
            if len(found_sensors) == 0:
                st.markdown(f'<div class="error-box">❌ No recognised sensor columns found. Expected columns like: '
                            f'<code>{", ".join(SURVIVING_SENSORS[:5])}</code>…<br>'
                            f'Your CSV has: <code>{", ".join(rf_df.columns.tolist()[:8])}</code>…<br>'
                            f'Download the template above to see the correct column names.</div>', unsafe_allow_html=True)
            elif len(rf_df) < 30:
                st.markdown(f'<div class="error-box">❌ Need at least <b>30 rows</b> for 30-cycle rolling features, found <b>{len(rf_df)}</b>.<br>'
                            f'Add more rows to your CSV. Download the template (40 rows) for reference.</div>', unsafe_allow_html=True)
            elif not rf_ok:
                st.markdown('<div class="error-box">❌ Model not loaded. Cannot run inference.</div>', unsafe_allow_html=True)
            else:
                missing_sensors = [c for c in SURVIVING_SENSORS if c not in rf_df.columns]
                if missing_sensors:
                    st.markdown(f'<div class="info-box" style="font-size:12px;">⚠️ <b>Partial sensor set:</b> Missing columns: <code>{", ".join(missing_sensors)}</code>.<br>'
                                f'The model will use only the {len(found_sensors)} available sensors. '
                                f'Missing sensors will be set to 0 in the feature vector, which may affect prediction accuracy.</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="success-box">✅ Found {len(found_sensors)} sensor columns: <code>{", ".join(found_sensors)}</code></div>', unsafe_allow_html=True)
                with st.spinner("🧠 Computing rolling features & running RF inference…"):
                    X_row, feat_cols = prepare_rf_xgb_row(rf_df, rf_model)
                    rul_pred_rf = float(np.clip(rf_model.predict(X_row)[0], 0, 125))

                st.markdown(f'<div class="success-box">✅ Feature vector shape: <b>(1, {len(feat_cols)})</b> — using {len(feat_cols)} features.</div>', unsafe_allow_html=True)
                render_prediction_result(rul_pred_rf, 125, model_label="Random Forest")

                with st.expander("📋 Feature vector used (last row after rolling)", expanded=False):
                    st.dataframe(X_row.T.rename(columns={X_row.index[-1]: "Value"}), use_container_width=True)

        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error: {e}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:40px 20px;border:2px dashed rgba(11,61,145,0.4);border-radius:14px;margin-top:20px;">
          <div style="font-size:48px;margin-bottom:14px;">🌲</div>
          <div style="font-size:17px;font-weight:600;color:{NASA_WHITE};margin-bottom:8px;">No file uploaded yet</div>
          <div style="font-size:13px;color:{NASA_GREY};max-width:480px;margin:0 auto;">
            Use the <b>Try a Sample Engine</b> panel above to run the model instantly,
            or upload your own CSV (≥ 30 rows, 12 sensors + unit_id, cycle, op_condition).
          </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — XGBOOST
# ════════════════════════════════════════════════════════════════════════════
with tab_xgb:
    st.markdown("### 🔥 XGBoost — Supervised Regression")
    st.markdown(f"""
    <div class="info-box">
      XGBoost uses the <b>same flat-feature format as Random Forest</b>.
      Upload a CSV of one engine's full sensor history (≥ 30 rows).
      Rolling statistics (30-cycle window) are computed automatically and
      the last row is used for inference.<br><br>
      <b>Required columns:</b> <code>unit_id</code>, <code>cycle</code>,
      the 12 surviving sensor columns (<code>sensor_01, 02, 03, 04, 07, 08, 09, 12, 13, 14, 17, 18</code>),
      and <code>op_condition</code> (integer 0–5). Values must be <b>StandardScaler-normalised</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Template download ────────────────────────────────────────────────────
    with st.expander("📥 Download CSV Template for XGBoost", expanded=False):
        st.markdown(f"""
        <div class="template-box">
          <b>Identical format to Random Forest</b> — XGBoost uses the exact same feature engineering pipeline.<br><br>
          • <b>45 rows</b> of simulated engine history (minimum 30 needed)<br>
          • Columns: <code>unit_id</code>, <code>cycle</code>, 12 surviving sensor columns, <code>op_condition</code><br>
          • Sensor columns: <code>sensor_01, 02, 03, 04, 07, 08, 09, 12, 13, 14, 17, 18</code><br>
          • Values: StandardScaler-normalised (~−3 to +3 range)<br><br>
          <b>How to use:</b> Fill in your normalised sensor readings and upload below.
          The app computes rolling stats automatically.
        </div>
        """, unsafe_allow_html=True)
        xgb_template = make_rf_xgb_template(45)
        st.dataframe(xgb_template.head(5), use_container_width=True)
        st.caption("Preview of first 5 rows. Template has 45 rows total.")
        st.download_button(
            label="⬇️ Download XGBoost Template CSV",
            data=df_to_csv_bytes(xgb_template),
            file_name="xgb_template.csv",
            mime="text/csv",
            key="xgb_template_dl"
        )

    xgb_ok = False
    try:
        with st.spinner("🔄 Loading XGBoost…"):
            xgb_model = load_xgb()
        xgb_ok = True
        st.markdown('<div class="success-box">✅ XGBoost loaded — ready for inference.</div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown(f'<div class="error-box">⚠️ <code>xgb_best_model.pkl</code> not found in <code>./lstm/</code>.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ Error loading XGBoost: {e}</div>', unsafe_allow_html=True)

    # ── Sample-engine scenario panel ─────────────────────────────────────────
    def _run_xgb(df_sc):
        if "unit_id" not in df_sc.columns:
            df_sc = df_sc.copy(); df_sc.insert(0, "unit_id", 1)
        if "cycle" not in df_sc.columns:
            df_sc = df_sc.copy(); df_sc.insert(1, "cycle", range(1, len(df_sc)+1))
        X_row, _ = prepare_rf_xgb_row(df_sc, xgb_model)
        return float(np.clip(xgb_model.predict(X_row)[0], 0, 125))

    render_scenario_panel("xgb", xgb_ok, _run_xgb)

    st.markdown("---")
    st.markdown("### 📁 Or Upload Your Own CSV")
    col_up3, col_hint3 = st.columns([2, 1])
    with col_up3:
        xgb_upload = st.file_uploader("📁 Upload engine sensor CSV (XGBoost)", type=["csv"], key="xgb_upload")
    with col_hint3:
        st.markdown(f"""
        <div style="padding-top:8px;">
        <div class="info-box" style="font-size:12px;">
          <b>Required columns:</b><br>
          • unit_id (engine ID)<br>
          • cycle (sequential int)<br>
          • sensor_01, 02, 03, 04, 07, 08, 09, 12, 13, 14, 17, 18<br>
          • op_condition (int 0–5)<br>
          • ≥ 30 rows needed<br>
          • Values: StandardScaled (~−3 to +3)<br>
          • Same format as Random Forest
        </div></div>""", unsafe_allow_html=True)

    if xgb_upload is not None:
        try:
            xgb_df = pd.read_csv(xgb_upload)
            st.markdown(f"**Preview** — {len(xgb_df)} rows × {xgb_df.shape[1]} columns")
            with st.expander("📊 Data preview (first 10 rows)", expanded=False):
                st.dataframe(xgb_df.head(10), use_container_width=True)

            if "unit_id" not in xgb_df.columns:
                xgb_df.insert(0, "unit_id", 1)
                st.markdown('<div class="info-box" style="font-size:12px;">ℹ️ <code>unit_id</code> column not found — defaulted to 1.</div>', unsafe_allow_html=True)
            if "cycle" not in xgb_df.columns:
                xgb_df.insert(1, "cycle", range(1, len(xgb_df)+1))
                st.markdown('<div class="info-box" style="font-size:12px;">ℹ️ <code>cycle</code> column not found — auto-generated as 1…N.</div>', unsafe_allow_html=True)

            found_sensors_xgb = [c for c in SURVIVING_SENSORS if c in xgb_df.columns]
            if len(found_sensors_xgb) == 0:
                st.markdown(f'<div class="error-box">❌ No recognised sensor columns found. Expected: '
                            f'<code>{", ".join(SURVIVING_SENSORS[:5])}</code>…<br>'
                            f'Download the template above to see the correct column names.</div>', unsafe_allow_html=True)
            elif len(xgb_df) < 30:
                st.markdown(f'<div class="error-box">❌ Need at least <b>30 rows</b> for rolling features, found <b>{len(xgb_df)}</b>.</div>', unsafe_allow_html=True)
            elif not xgb_ok:
                st.markdown('<div class="error-box">❌ Model not loaded. Cannot run inference.</div>', unsafe_allow_html=True)
            else:
                missing_sensors_xgb = [c for c in SURVIVING_SENSORS if c not in xgb_df.columns]
                if missing_sensors_xgb:
                    st.markdown(f'<div class="info-box" style="font-size:12px;">⚠️ <b>Partial sensor set:</b> Missing columns: <code>{", ".join(missing_sensors_xgb)}</code>.<br>'
                                f'The model will use only the {len(found_sensors_xgb)} available sensors. '
                                f'Missing sensors default to 0 in the feature vector, which may shift the prediction.</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="success-box">✅ Found {len(found_sensors_xgb)} sensor columns.</div>', unsafe_allow_html=True)
                with st.spinner("🧠 Computing rolling features & running XGBoost inference…"):
                    X_row_xgb, feat_cols_xgb = prepare_rf_xgb_row(xgb_df, xgb_model)
                    rul_pred_xgb = float(np.clip(xgb_model.predict(X_row_xgb)[0], 0, 125))

                st.markdown(f'<div class="success-box">✅ Feature vector shape: <b>(1, {len(feat_cols_xgb)})</b>.</div>', unsafe_allow_html=True)
                render_prediction_result(rul_pred_xgb, 125, model_label="XGBoost")

                with st.expander("📋 Feature vector used (last row after rolling)", expanded=False):
                    st.dataframe(X_row_xgb.T.rename(columns={X_row_xgb.index[-1]: "Value"}), use_container_width=True)

        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error: {e}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:40px 20px;border:2px dashed rgba(11,61,145,0.4);border-radius:14px;margin-top:20px;">
          <div style="font-size:48px;margin-bottom:14px;">🔥</div>
          <div style="font-size:17px;font-weight:600;color:{NASA_WHITE};margin-bottom:8px;">No file uploaded yet</div>
          <div style="font-size:13px;color:{NASA_GREY};max-width:480px;margin:0 auto;">
            Use the <b>Try a Sample Engine</b> panel above to run the model instantly,
            or upload your own CSV (same format as Random Forest).
          </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — K-MEANS EXPLORER (educational / static visualisation only)
# ════════════════════════════════════════════════════════════════════════════
with tab_km:
    st.markdown("### 🔵 K-Means Clustering — Unsupervised Health-State Explorer")
    st.markdown(f"""
    <div class="info-box">
      <b>This tab is an educational explorer</b>, not a prediction tool. It shows what K-Means learned
      from the 249 training engines in FD004 — how engines naturally group into health states based on
      their degradation fingerprints (mean sensor readings over the last 30 cycles).<br><br>
      Use the controls below to explore different values of K and see how cluster quality and 
      interpretability change. The visualisations reveal the natural structure of turbofan engine degradation.
      <b>No new data upload is required or supported here.</b>
    </div>
    """, unsafe_allow_html=True)

    train_df_km = load_train_data()

    if train_df_km is None:
        st.markdown(f"""
        <div class="error-box">
          ⚠️ <code>train_regression.csv</code> not found in <code>./lstm/</code>.<br>
          This file is needed to run the K-Means explorer. Run Part 7 of the notebook to export it.
        </div>""", unsafe_allow_html=True)
    else:
        # ── K selector ──────────────────────────────────────────────────────
        sensor_cols_km = [c for c in SENSOR_COLS_ALL if c in train_df_km.columns]

        if len(sensor_cols_km) == 0:
            st.markdown('<div class="error-box">❌ No sensor columns found in train_regression.csv.</div>', unsafe_allow_html=True)
        else:
            # Build per-engine fingerprint (mean of last 30 cycles)
            @st.cache_data(show_spinner=False)
            def build_fingerprints(sensor_cols):
                last30 = train_df_km.groupby("unit_id").apply(
                    lambda g: g.sort_values("cycle").tail(30)[sensor_cols].mean()
                ).reset_index()
                return last30

            fp_df = build_fingerprints(sensor_cols_km)
            X_fp_raw = fp_df[sensor_cols_km].values

            # MinMax scale (matching notebook's approach)
            from sklearn.preprocessing import MinMaxScaler
            mm = MinMaxScaler()
            X_fp = mm.fit_transform(X_fp_raw)

            # Also load RUL info for each engine (last row)
            rul_per_engine = (
                train_df_km.sort_values("cycle")
                .groupby("unit_id", as_index=False)
                .tail(1)[["unit_id", "RUL_clipped"]]
                if "RUL_clipped" in train_df_km.columns
                else None
            )

            cycle_per_engine = (
                train_df_km.groupby("unit_id")["cycle"].max()
                .reset_index()
                .rename(columns={"cycle": "max_cycle"})
            )

            st.markdown("---")
            st.markdown("#### ⚙️ Clustering Parameters")


            col_k, col_info = st.columns([1, 2])
            with col_k:
                k_val = st.slider(
                    "Number of Clusters (K)", min_value=2, max_value=5, value=2, step=1,
                    help="K=2 matches elbow & silhouette optimal. Try higher values to see finer health stages."
                )

            # ── Compute elbow + silhouette across k=2..8 ────────────────────
            @st.cache_data(show_spinner=False)
            def compute_cluster_metrics(X):
                inertias, sils = [], []
                K_range = range(2, 6)
                for k in K_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labs = km.fit_predict(X)
                    inertias.append(km.inertia_)
                    sils.append(silhouette_score(X, labs))
                return list(K_range), inertias, sils

            k_range, inertias, sils = compute_cluster_metrics(X_fp)
            best_k_idx = int(np.argmax(sils))
            best_k = k_range[best_k_idx]

            with col_info:
                st.markdown(f"""
                <div class="info-box" style="font-size:13px;">
                  <b>Optimal K from silhouette score: K = {best_k}</b><br>
                  Silhouette at K={best_k}: <b>{sils[best_k_idx]:.3f}</b> (range −1 to 1, higher = better separation)<br>
                  Currently selected: K = {k_val} &nbsp;|&nbsp; Silhouette: <b>{sils[k_val-2]:.3f}</b>
                </div>""", unsafe_allow_html=True)

            # ── Fit K-Means with selected K ──────────────────────────────────
            @st.cache_data(show_spinner=False)
            def fit_kmeans(X, k):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                sil = silhouette_score(X, labels)
                db  = davies_bouldin_score(X, labels)
                ch  = calinski_harabasz_score(X, labels)
                return km, labels, sil, db, ch

            km_fit, km_labels, sil_val, db_val, ch_val = fit_kmeans(X_fp, k_val)

            # Assign health-stage names based on mean sensor level per cluster
            cluster_means_val = pd.Series(km_fit.cluster_centers_.mean(axis=1))
            # Order from lowest (healthiest) to highest (most degraded)
            order = cluster_means_val.argsort().values  # cluster IDs sorted by health
            health_stage_names = ["Healthy", "Early Degradation", "Moderate Degradation",
                                  "Severe Degradation", "Critical", "Stage 6", "Stage 7", "Stage 8"]
            stage_map = {cluster_id: health_stage_names[rank] for rank, cluster_id in enumerate(order)}
            stage_colours = ["#00C853", "#76FF03", "#FFA500", "#FF3D00", "#B71C1C", "#7B1FA2", "#0D47A1", "#006064"]
            colour_map = {cluster_id: stage_colours[rank] for rank, cluster_id in enumerate(order)}

            fp_df["cluster"] = km_labels
            fp_df["health_stage"] = fp_df["cluster"].map(stage_map)
            fp_df["colour"] = fp_df["cluster"].map(colour_map)

            # ── Quality metrics row ──────────────────────────────────────────
            st.markdown("---")
            st.markdown(f"#### 📐 Cluster Quality Metrics — K = {k_val}")
            m1, m2, m3, m4 = st.columns(4)
            metric_data = [
                ("Silhouette Score", f"{sil_val:.3f}", "Higher = better (max 1.0)", sil_val > 0.35),
                ("Davies-Bouldin", f"{db_val:.3f}", "Lower = better (compact clusters)", db_val < 1.0),
                ("Calinski-Harabasz", f"{ch_val:.0f}", "Higher = better (separation)", ch_val > 200),
                ("Engines Clustered", "249", "All training engines", True),
            ]
            for col, (label, val, desc, _good) in zip([m1,m2,m3,m4], metric_data):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">{label}</div>
                      <div class="metric-value" style="font-size:28px;">{val}</div>
                      <div class="metric-unit">{desc}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Sub-tabs for visualisations ──────────────────────────────────
            km_v1, km_v2, km_v3, km_v4, km_v5 = st.tabs([
                "📈 Elbow & Silhouette", "🗺️ PCA Projection",
                "📊 Cluster Profiles", "🔬 Sensor Breakdown", "🔄 Engine Lifecycle"
            ])

            # Elbow + Silhouette
            with km_v1:
                st.markdown("#### Elbow & Silhouette — Choosing Optimal K")
                st.markdown(f"""
                <div class="info-box" style="font-size:13px;">
                  These two curves are used together to pick the best K.
                  The <b>elbow</b> shows where adding more clusters gives diminishing returns in compactness.
                  The <b>silhouette score</b> directly measures how well-separated clusters are — peak = optimal K.
                  Both methods agree on <b>K = {best_k}</b> for this dataset.
                </div>""", unsafe_allow_html=True)

                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=k_range, y=inertias, mode="lines+markers", name="Inertia (Elbow)",
                    line=dict(color="#00B4D8", width=2), marker=dict(size=8)
                ))
                # When best_k == k_val, draw one combined line to avoid overlap
                if best_k == k_val:
                    fig_elbow.add_vline(x=best_k, line_dash="dash", line_color=NASA_RED,
                                        annotation_text=f"Optimal & Selected K={best_k}",
                                        annotation_font_color=NASA_WHITE,
                                        annotation_position="top right")
                else:
                    fig_elbow.add_vline(x=best_k, line_dash="dash", line_color=NASA_RED,
                                        annotation_text=f"Optimal K={best_k}",
                                        annotation_font_color=NASA_WHITE,
                                        annotation_position="top right")
                    fig_elbow.add_vline(x=k_val, line_dash="dot", line_color="#FFA500",
                                        annotation_text=f"Selected K={k_val}",
                                        annotation_font_color="#FFA500",
                                        annotation_position="top left")
                fig_elbow.update_layout(
                    title="Elbow Curve (Inertia vs K)", xaxis_title="K (Number of Clusters)",
                    yaxis_title="Inertia (Within-cluster SSE)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color=NASA_WHITE, height=320
                )

                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(
                    x=k_range, y=sils, mode="lines+markers", name="Silhouette",
                    line=dict(color="#00C853", width=2), marker=dict(size=8)
                ))
                if best_k == k_val:
                    fig_sil.add_vline(x=best_k, line_dash="dash", line_color=NASA_RED,
                                      annotation_text=f"Optimal & Selected K={best_k}",
                                      annotation_font_color=NASA_WHITE,
                                      annotation_position="top right")
                else:
                    fig_sil.add_vline(x=best_k, line_dash="dash", line_color=NASA_RED,
                                      annotation_text=f"Best K={best_k}",
                                      annotation_font_color=NASA_WHITE,
                                      annotation_position="top right")
                    fig_sil.add_vline(x=k_val, line_dash="dot", line_color="#FFA500",
                                      annotation_text=f"Selected K={k_val}",
                                      annotation_font_color="#FFA500",
                                      annotation_position="top left")
                fig_sil.update_layout(
                    title="Silhouette Score vs K", xaxis_title="K",
                    yaxis_title="Silhouette Score (higher = better)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color=NASA_WHITE, height=320
                )

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.plotly_chart(fig_elbow, use_container_width=True)
                with col_e2:
                    st.plotly_chart(fig_sil, use_container_width=True)

            # PCA Projection
            with km_v2:
                st.markdown("#### PCA 2D Projection of Engine Degradation Fingerprints")
                pca_km = PCA(n_components=2, random_state=42)
                X_2d = pca_km.fit_transform(X_fp)
                ev = pca_km.explained_variance_ratio_ * 100

                centers_2d = pca_km.transform(km_fit.cluster_centers_)

                st.markdown(f"""
                <div class="info-box" style="font-size:13px;">
                  <b>PC1 explains {ev[0]:.1f}% of variance</b> — this single axis captures most of the engine 
                  degradation signal. All 12 sensors co-degrade along this dominant direction, which is why 
                  K=2 (Healthy vs Degraded) is the natural structure. Each point is one engine's 
                  degradation fingerprint (mean of its last 30 sensor cycles).
                </div>""", unsafe_allow_html=True)

                fig_pca = go.Figure()
                for cid in sorted(fp_df["cluster"].unique()):
                    mask = fp_df["cluster"] == cid
                    stage = stage_map[cid]
                    clr = colour_map[cid]
                    fig_pca.add_trace(go.Scatter(
                        x=X_2d[mask, 0], y=X_2d[mask, 1],
                        mode="markers", name=f"Cluster {cid}: {stage}",
                        marker=dict(size=9, color=clr, opacity=0.8,
                                    line=dict(color="white", width=0.5)),
                        text=[f"Engine {uid}" for uid in fp_df.loc[mask, "unit_id"]],
                        hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>"
                    ))

                fig_pca.add_trace(go.Scatter(
                    x=centers_2d[:, 0], y=centers_2d[:, 1],
                    mode="markers", name="Centroids",
                    marker=dict(size=16, color="white", symbol="x",
                                line=dict(color="black", width=2))
                ))
                fig_pca.update_layout(
                    title=f"K-Means (K={k_val}) — PCA 2D Projection of 249 Engines",
                    xaxis_title=f"PC1 ({ev[0]:.1f}% variance explained)",
                    yaxis_title=f"PC2 ({ev[1]:.1f}% variance explained)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color=NASA_WHITE, legend=dict(font=dict(color=NASA_WHITE)),
                    height=500
                )
                st.plotly_chart(fig_pca, use_container_width=True)

            # Cluster Profiles
            with km_v3:
                st.markdown("#### Cluster Profiles — Average Characteristics per Health Stage")

                # Merge fingerprints with RUL and cycle info
                profile_df = fp_df[["unit_id", "cluster", "health_stage"]].copy()
                if rul_per_engine is not None:
                    profile_df = profile_df.merge(rul_per_engine, on="unit_id", how="left")
                profile_df = profile_df.merge(cycle_per_engine, on="unit_id", how="left")

                summary_rows = []
                for cid in sorted(fp_df["cluster"].unique()):
                    mask = profile_df["cluster"] == cid
                    row = {
                        "Cluster": cid,
                        "Health Stage": stage_map[cid],
                        "Engine Count": int(mask.sum()),
                    }
                    if "RUL_clipped" in profile_df.columns:
                        row["Avg RUL (cycles)"] = round(profile_df.loc[mask, "RUL_clipped"].mean(), 1)
                        row["Min RUL"] = round(profile_df.loc[mask, "RUL_clipped"].min(), 1)
                        row["Max RUL"] = round(profile_df.loc[mask, "RUL_clipped"].max(), 1)
                    row["Avg Lifetime (cycles)"] = round(profile_df.loc[mask, "max_cycle"].mean(), 1)
                    row["Centroid Mean (scaled)"] = round(float(km_fit.cluster_centers_[cid].mean()), 4)
                    summary_rows.append(row)

                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # Bar chart: engine count per cluster
                fig_count = go.Figure(go.Bar(
                    x=[f"Cluster {r['Cluster']}: {r['Health Stage']}" for r in summary_rows],
                    y=[r["Engine Count"] for r in summary_rows],
                    marker_color=[colour_map[r["Cluster"]] for r in summary_rows],
                    text=[r["Engine Count"] for r in summary_rows],
                    textposition="outside", textfont=dict(color=NASA_WHITE)
                ))
                fig_count.update_layout(
                    title="Number of Engines per Health Stage",
                    xaxis_title="Health Stage", yaxis_title="Engine Count",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color=NASA_WHITE, height=360
                )
                st.plotly_chart(fig_count, use_container_width=True)

                # RUL distribution per cluster
                if "RUL_clipped" in profile_df.columns:
                    fig_rul_box = go.Figure()
                    for cid in sorted(fp_df["cluster"].unique()):
                        mask = profile_df["cluster"] == cid
                        fig_rul_box.add_trace(go.Box(
                            y=profile_df.loc[mask, "RUL_clipped"],
                            name=f"Cluster {cid}: {stage_map[cid]}",
                            marker_color=colour_map[cid],
                            boxmean=True
                        ))
                    fig_rul_box.update_layout(
                        title="RUL Distribution by Health Stage (at engine end-of-record)",
                        yaxis_title="RUL (cycles)", xaxis_title="Health Stage",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color=NASA_WHITE, height=400
                    )
                    st.plotly_chart(fig_rul_box, use_container_width=True)
                    st.markdown(f"""
                    <div class="info-box" style="font-size:13px;">
                      Each box shows the RUL distribution at the <b>last recorded cycle</b> for engines in that cluster.
                      The healthy cluster tends toward higher RUL values; the degraded cluster toward lower.
                      This validates that the unsupervised clustering separates engines by actual health state —
                      even though RUL labels were <b>never used</b> as input to K-Means.
                    </div>""", unsafe_allow_html=True)

            # Sensor Breakdown
            with km_v4:
                st.markdown("#### Sensor-Level Comparison Across Clusters")
                st.markdown(f"""
                <div class="info-box" style="font-size:13px;">
                  This heatmap shows the mean MinMax-scaled sensor value per cluster centroid.
                  Sensors that differ most between clusters are the strongest drivers of health-state separation.
                  <b>sensor_03, sensor_17, sensor_04</b> typically show the largest inter-cluster difference —
                  these correspond to high-pressure compressor temperature and pressure readings,
                  which degrade most visibly in turbofan engines.
                </div>""", unsafe_allow_html=True)

                centroid_df = pd.DataFrame(
                    km_fit.cluster_centers_,
                    columns=sensor_cols_km,
                    index=[f"Cluster {i}: {stage_map[i]}" for i in range(k_val)]
                )

                fig_heat = go.Figure(go.Heatmap(
                    z=centroid_df.values,
                    x=sensor_cols_km,
                    y=centroid_df.index,
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar={"tickfont": {"color": NASA_GREY}, "title": {"text": "Scaled Value", "font": {"color": NASA_GREY}}},
                    hovertemplate="Sensor: %{x}<br>Stage: %{y}<br>Value: %{z:.3f}<extra></extra>"
                ))
                fig_heat.update_layout(
                    title=f"Cluster Centroid Heatmap — K={k_val} (MinMax Scaled Sensors)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color=NASA_WHITE, height=max(300, 80 + k_val * 60),
                    xaxis={"tickfont": {"color": NASA_GREY, "size": 10}, "tickangle": -30},
                    yaxis={"tickfont": {"color": NASA_GREY, "size": 11}}
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                # Radar chart comparing all clusters on top sensors
                st.markdown("#### Radar Chart — Cluster Centroids vs Sensors")
                n_show = min(10, len(sensor_cols_km))
                sensors_show = sensor_cols_km[:n_show]

                # Sort by inter-cluster variance to show most discriminating sensors
                inter_var = centroid_df[sensors_show].std()
                sensors_show_sorted = inter_var.sort_values(ascending=False).index.tolist()

                fig_radar = go.Figure()
                for cid in range(k_val):
                    vals = centroid_df.iloc[cid][sensors_show_sorted].values.tolist()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=sensors_show_sorted + [sensors_show_sorted[0]],
                        fill="toself",
                        name=f"Cluster {cid}: {stage_map[cid]}",
                        line_color=colour_map[cid],
                        opacity=0.7
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, color=NASA_GREY)),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color=NASA_WHITE, legend=dict(font=dict(color=NASA_WHITE)),
                    height=440, title="Sensor Profile per Cluster (sorted by discriminative power)"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Engine Lifecycle / Progression
            with km_v5:
                st.markdown("#### Engine Lifecycle Progression Through Health Stages")
                st.markdown(f"""
                <div class="info-box" style="font-size:13px;">
                  This view shows how individual engines progress through sensor space over their lifecycle.
                  Select an engine and see where it falls in the PCA space relative to the cluster centroids.
                  Engines that die quickly tend to start closer to the degraded centroid even early in life.
                </div>""", unsafe_allow_html=True)

                if "cycle" in train_df_km.columns:
                    units_avail = sorted(train_df_km["unit_id"].unique())
                    sel_units = st.multiselect(
                        "Select engines to trace (max 5):", units_avail,
                        default=units_avail[:3], max_selections=5
                    )

                    if sel_units:
                        fig_trace = go.Figure()

                        # Plot all engine fingerprints as background
                        fig_trace.add_trace(go.Scatter(
                            x=X_2d[:, 0], y=X_2d[:, 1],
                            mode="markers", name="All Engines (fingerprint)",
                            marker=dict(size=5, color="rgba(100,130,180,0.3)"),
                            showlegend=True
                        ))

                        # Plot centroids
                        for cid in range(k_val):
                            fig_trace.add_trace(go.Scatter(
                                x=[centers_2d[cid, 0]], y=[centers_2d[cid, 1]],
                                mode="markers+text",
                                name=f"Centroid {cid}: {stage_map[cid]}",
                                marker=dict(size=18, color=colour_map[cid],
                                            symbol="x", line=dict(color="white", width=2)),
                                text=[stage_map[cid]], textposition="top center",
                                textfont=dict(color=colour_map[cid], size=11)
                            ))

                        # Trace selected engines' sensor trajectory over cycles
                        colours_trace = ["#00B4D8", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]
                        for i, uid in enumerate(sel_units):
                            unit_data = train_df_km[train_df_km["unit_id"] == uid].sort_values("cycle")
                            clr = colours_trace[i % len(colours_trace)]

                            # Compute rolling window PCA coordinates over lifecycle
                            window = 30
                            points_x, points_y, point_cycles = [], [], []
                            for c_idx in range(window, len(unit_data)+1, max(1, len(unit_data)//10)):
                                window_data = unit_data.iloc[max(0, c_idx-window):c_idx][sensor_cols_km]
                                mean_row = window_data.mean().values.reshape(1,-1)
                                mean_scaled = mm.transform(mean_row)
                                pt_2d = pca_km.transform(mean_scaled)
                                points_x.append(pt_2d[0,0])
                                points_y.append(pt_2d[0,1])
                                point_cycles.append(int(unit_data.iloc[min(c_idx-1, len(unit_data)-1)]["cycle"]))

                            if points_x:
                                fig_trace.add_trace(go.Scatter(
                                    x=points_x, y=points_y,
                                    mode="lines+markers",
                                    name=f"Engine {uid} trajectory",
                                    line=dict(color=clr, width=2, dash="dot"),
                                    marker=dict(size=7, color=clr),
                                    text=[f"Engine {uid}, Cycle {c}" for c in point_cycles],
                                    hovertemplate="<b>%{text}</b><extra></extra>"
                                ))
                                # Mark start and end
                                if len(points_x) > 1:
                                    fig_trace.add_trace(go.Scatter(
                                        x=[points_x[0], points_x[-1]],
                                        y=[points_y[0], points_y[-1]],
                                        mode="markers",
                                        name=f"Engine {uid} start→end",
                                        marker=dict(size=12, color=clr,
                                                    symbol=["circle", "star"],
                                                    line=dict(color="white", width=1.5)),
                                        showlegend=False
                                    ))

                        fig_trace.update_layout(
                            title="Engine Trajectories Through Degradation Space (PCA projection)",
                            xaxis_title=f"PC1 ({ev[0]:.1f}% variance)",
                            yaxis_title=f"PC2 ({ev[1]:.1f}% variance)",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color=NASA_WHITE, legend=dict(font=dict(color=NASA_WHITE, size=10)),
                            height=560
                        )
                        st.plotly_chart(fig_trace, use_container_width=True)
                        st.markdown(f"""
                        <div class="info-box" style="font-size:12px;">
                          ⭕ = engine start &nbsp;|&nbsp; ⭐ = engine end (failure) &nbsp;|&nbsp;
                          Dotted lines show the trajectory through PCA space as the engine degrades.
                          Engines moving toward the higher-wear centroid are degrading faster.
                        </div>""", unsafe_allow_html=True)

        # Always-shown methodology note
        with st.expander("📋 K-Means Methodology & Why No Prediction", expanded=False):
            st.markdown("""
**Why this tab doesn't predict RUL or classify new engines:**

K-Means is an *unsupervised* algorithm — it finds natural groupings in existing data without using labels.
Its value here is *descriptive*, not predictive: it tells us "engines in the training set naturally fall into
these health stages". For actual RUL prediction on new engine data, use the Random Forest or XGBoost tabs.

**Input Feature:** Mean of each sensor's last 30 cycles → one row per engine (degradation fingerprint).

**Optimal K:** Selected as K=2 by both Elbow curve and Silhouette score.
Silhouette ≈ 0.43 | Davies-Bouldin ≈ 0.85 | Calinski-Harabasz ≈ 291

**PCA Analysis:** PC1 explains ~80% of variance — all sensors co-degrade along a single dominant axis.
This confirms that K=2 (Healthy vs Degraded) is the natural structure of the data.

**Key sensors driving separation:** sensor_03, sensor_17, sensor_04
(temperature and pressure at the high-pressure compressor — the first to degrade in turbofan engines).

**Validation:** The clusters correlate strongly with RUL (even though RUL was never used as input),
confirming that sensor degradation patterns are sufficient for unsupervised health-state discovery.
            """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — EDA / DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("### 📊 Exploratory Data Analysis — NASA C-MAPSS FD004")
    st.markdown(f"""
    <div class="info-box">
      Interactive visualisations of the FD004 dataset and preprocessing pipeline.
      Load <code>train_regression.csv</code> and/or <code>test_predictions.csv</code>
      from the <code>./lstm/</code> folder to unlock all charts.
    </div>
    """, unsafe_allow_html=True)

    train_df = load_train_data()
    pred_df  = load_test_predictions()

    if train_df is None and pred_df is None:
        st.markdown(f"""
        <div class="error-box">
          ⚠️ Neither <code>train_regression.csv</code> nor <code>test_predictions.csv</code>
          found in <code>./lstm/</code>.<br>
          Run Part 7 of the notebook to export Streamlit assets, then copy them into <code>./lstm/</code>.
        </div>""", unsafe_allow_html=True)
    else:
        eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
            "🔬 Engine Lifetimes", "📉 Sensor Trends", "🎯 RUL Distribution",
            "🔵 Clustering (PCA)", "📐 Preprocessing Pipeline"
        ])

        # ── EDA: Engine Lifetimes ───────────────────────────────────────────
        with eda_tab1:
            if train_df is not None:
                st.markdown("#### Engine Lifetime Distribution (Training Set)")
                unit_life = train_df.groupby("unit_id")["cycle"].max().sort_values()
                c1, c2 = st.columns(2)
                with c1:
                    fig = go.Figure(go.Bar(x=unit_life.index, y=unit_life.values,
                                           marker_color=NASA_BLUE, opacity=0.8))
                    fig.update_layout(title="Lifetime per Engine Unit",
                                      xaxis_title="Engine ID", yaxis_title="Total Cycles",
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color=NASA_WHITE, height=350)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig2 = go.Figure(go.Histogram(x=unit_life.values, nbinsx=30,
                                                  marker_color=NASA_BLUE, opacity=0.8))
                    fig2.add_vline(x=unit_life.mean(), line_dash="dash", line_color=NASA_RED,
                                   annotation_text=f"Mean={unit_life.mean():.0f}", annotation_font_color=NASA_WHITE)
                    fig2.update_layout(title="Lifetime Distribution", xaxis_title="Total Cycles",
                                       yaxis_title="Frequency",
                                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font_color=NASA_WHITE, height=350)
                    st.plotly_chart(fig2, use_container_width=True)

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                for col, label, val in zip([col_m1, col_m2, col_m3, col_m4],
                                            ["Engines", "Max Lifetime", "Min Lifetime", "Mean Lifetime"],
                                            [train_df["unit_id"].nunique(), unit_life.max(),
                                             unit_life.min(), f"{unit_life.mean():.1f}"]):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="metric-label">{label}</div>
                          <div class="metric-value" style="font-size:28px;">{val}</div>
                        </div>""", unsafe_allow_html=True)

        # ── EDA: Sensor Trends ──────────────────────────────────────────────
        with eda_tab2:
            if train_df is not None:
                st.markdown("#### Sensor Trends Over Engine Lifecycle")
                sensor_cols_avail = [c for c in SENSOR_COLS_ALL if c in train_df.columns]
                units_avail = sorted(train_df["unit_id"].unique())
                c1, c2 = st.columns(2)
                with c1:
                    sel_unit = st.selectbox("Select Engine Unit", units_avail, index=0)
                with c2:
                    sel_sensors = st.multiselect("Select Sensors", sensor_cols_avail,
                                                  default=sensor_cols_avail[:4])
                if sel_sensors:
                    unit_data = train_df[train_df["unit_id"] == sel_unit].sort_values("cycle")
                    fig = go.Figure()
                    colours = px.colors.qualitative.Plotly
                    for i, s in enumerate(sel_sensors):
                        if s in unit_data.columns:
                            fig.add_trace(go.Scatter(x=unit_data["cycle"], y=unit_data[s],
                                                      mode="lines", name=s,
                                                      line=dict(color=colours[i % len(colours)], width=1.5)))
                    fig.update_layout(title=f"Sensor Trends — Engine Unit {sel_unit}",
                                      xaxis_title="Cycle", yaxis_title="Sensor Value",
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color=NASA_WHITE, legend=dict(font=dict(color=NASA_WHITE)),
                                      height=420)
                    st.plotly_chart(fig, use_container_width=True)

                # Sensor std bar chart
                st.markdown("#### Sensor Variance (Std Dev) — Identifying Constant Sensors")
                std_vals = train_df[sensor_cols_avail].std().sort_values(ascending=False)
                fig_std = go.Figure(go.Bar(x=std_vals.index, y=std_vals.values,
                                           marker_color=[NASA_RED if v < 10 else "#00B4D8" for v in std_vals.values]))
                fig_std.add_hline(y=10, line_dash="dash", line_color=NASA_RED,
                                  annotation_text="Removal threshold (std < 10)", annotation_font_color=NASA_WHITE)
                fig_std.update_layout(title="Standard Deviation per Sensor (Train Set)",
                                       xaxis_title="Sensor", yaxis_title="Std Dev",
                                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font_color=NASA_WHITE, height=350)
                st.plotly_chart(fig_std, use_container_width=True)

        # ── EDA: RUL Distribution ───────────────────────────────────────────
        with eda_tab3:
            if train_df is not None:
                st.markdown("#### RUL Distribution — Before & After Piecewise Clipping")
                rul_col = "RUL" if "RUL" in train_df.columns else ("RUL_clipped" if "RUL_clipped" in train_df.columns else None)
                clip_col = "RUL_clipped" if "RUL_clipped" in train_df.columns else None
                c1, c2 = st.columns(2)
                with c1:
                    if rul_col:
                        fig = go.Figure(go.Histogram(x=train_df[rul_col], nbinsx=60,
                                                     marker_color="#4E79A7", opacity=0.85))
                        fig.update_layout(title="RUL — Original", xaxis_title="RUL (cycles)",
                                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font_color=NASA_WHITE, height=320)
                        st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if clip_col:
                        fig2 = go.Figure(go.Histogram(x=train_df[clip_col], nbinsx=60,
                                                      marker_color="#F28E2B", opacity=0.85))
                        fig2.add_vline(x=125, line_dash="dash", line_color=NASA_RED,
                                       annotation_text="Cap = 125", annotation_font_color=NASA_WHITE)
                        fig2.update_layout(title="RUL — Clipped at 125", xaxis_title="RUL (cycles)",
                                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                           font_color=NASA_WHITE, height=320)
                        st.plotly_chart(fig2, use_container_width=True)

                # Operating conditions scatter
                if "op_condition" in train_df.columns:
                    st.markdown("#### Operating Condition Clusters (KMeans k=6)")
                    sensor_avail = [c for c in SENSOR_COLS_ALL if c in train_df.columns]
                    if len(sensor_avail) >= 2:
                        sample = train_df.sample(min(5000, len(train_df)), random_state=42)
                        fig_op = px.scatter(sample, x=sensor_avail[0], y=sensor_avail[1],
                                             color="op_condition", color_continuous_scale="Turbo",
                                             opacity=0.4, title="Sensors coloured by Operating Condition Cluster")
                        fig_op.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                              font_color=NASA_WHITE, height=400)
                        st.plotly_chart(fig_op, use_container_width=True)

        # ── EDA: PCA Clustering ─────────────────────────────────────────────
        with eda_tab4:
            if train_df is not None:
                st.markdown("#### PCA Projection of Engine Degradation Fingerprints")
                sensor_avail = [c for c in SENSOR_COLS_ALL if c in train_df.columns]
                if sensor_avail:
                    # Build per-engine fingerprint (mean of last 30 cycles)
                    last30 = train_df.groupby("unit_id").apply(
                        lambda g: g.sort_values("cycle").tail(30)[sensor_avail].mean()
                    ).reset_index()
                    X_fp_eda = last30[sensor_avail].values
                    pca = PCA(n_components=2, random_state=42)
                    X_2d_eda = pca.fit_transform(X_fp_eda)
                    ev_eda = pca.explained_variance_ratio_ * 100

                    color_col = None
                    try:
                        km_viz, _, _ = load_kmeans_artifacts()
                        mm_viz = MinMaxScaler()
                        X_fp_scaled = mm_viz.fit_transform(X_fp_eda)
                        n_feat_km = km_viz.cluster_centers_.shape[1]
                        X_fp_aligned = X_fp_scaled[:, :n_feat_km]
                        clusters_eda = km_viz.predict(X_fp_aligned)
                        last30["cluster"] = clusters_eda
                        color_col = "cluster"
                    except Exception:
                        pass

                    last30["PC1"] = X_2d_eda[:, 0]; last30["PC2"] = X_2d_eda[:, 1]

                    fig_pca = px.scatter(last30, x="PC1", y="PC2", color=color_col,
                                         hover_data=["unit_id"],
                                         color_continuous_scale="RdYlGn_r",
                                         title=f"Engine Fingerprints — PCA 2D  (PC1={ev_eda[0]:.1f}%  PC2={ev_eda[1]:.1f}%)")
                    fig_pca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                           font_color=NASA_WHITE, height=480,
                                           xaxis_title=f"PC1 ({ev_eda[0]:.1f}% var)",
                                           yaxis_title=f"PC2 ({ev_eda[1]:.1f}% var)")
                    st.plotly_chart(fig_pca, use_container_width=True)
                    st.markdown(f"""
                    <div class="info-box">
                      <b>PC1 explains {ev_eda[0]:.1f}% of variance</b> — all {len(sensor_avail)} sensors
                      co-degrade along a single dominant axis, confirming that K=2 (Healthy vs Degraded)
                      is the natural structure of this data.
                    </div>""", unsafe_allow_html=True)

        # ── EDA: Preprocessing pipeline ─────────────────────────────────────
        with eda_tab5:
            st.markdown("#### Preprocessing Pipeline Visualisation")
            st.markdown(f"""
            <div class="info-box">
              Step-by-step overview of how raw FD004 data was transformed before model training.
            </div>""", unsafe_allow_html=True)

            steps = [
                ("1️⃣", "Load Raw Data", "26 columns: unit_id, cycle, 3 op_settings, 21 sensors\n249 training engines | 248 test engines"),
                ("2️⃣", "Drop Constant Sensors", "Remove sensors with std < 10 across all cycles\nDropped: sensor_05, 06, 10, 11, 15, 16, 19, 20, 21 + op_set_3\nResult: 12 remaining sensor features (sensor_01,02,03,04,07,08,09,12,13,14,17,18)"),
                ("3️⃣", "Compute RUL Labels", "RUL = max_cycle(engine) − current_cycle\nApplied per engine in training set only"),
                ("4️⃣", "Clip RUL at 125 cycles", "Piecewise linear target: early cycles capped at 125\nFocuses model on the degradation phase, not early life"),
                ("5️⃣", "Rolling Mean Smoothing", "5-cycle rolling average per engine per sensor\nReduces high-frequency noise, preserves degradation trend"),
                ("6️⃣", "KMeans Operating Conditions", "k=6 clusters from (op_set_1, op_set_2)\nAssigns integer cluster label 0–5 to each row"),
                ("7️⃣", "Per-Condition StandardScaler", "StandardScaler fitted per condition on train only\nNormalises sensor readings within each operating regime"),
                ("8️⃣", "Sliding Window Sequences (LSTM)", "Window length = 30 cycles, stride = 1\nShapes: (N_sequences, 30, 13)"),
                ("9️⃣", "Rolling Stats (RF/XGBoost)", "30-cycle rolling mean + std per sensor\n12 sensors → 12 raw + 12 rolling means + 12 rolling stds + op_condition = 37 features total"),
            ]
            for icon, title, detail in steps:
                with st.expander(f"{icon} {title}", expanded=False):
                    for line in detail.split("\n"):
                        st.markdown(f"- {line}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### 📈 Model Comparison Dashboard")

    pred_df_cmp = load_test_predictions()

    if pred_df_cmp is not None and "RF_Predicted_RUL" in pred_df_cmp.columns:
        st.markdown(f'<div class="success-box">✅ Loaded test_predictions.csv — {len(pred_df_cmp)} rows.</div>', unsafe_allow_html=True)

        # Build per-engine last-row metrics
        pred_cols = {
            "Random Forest": "RF_Predicted_RUL",
            "XGBoost":       "XGB_Predicted_RUL",
            "CNN-LSTM":      "LSTM_Predicted_RUL",
        }
        available = {k: v for k, v in pred_cols.items() if v in pred_df_cmp.columns}

        # ── Decide which RUL column to use as ground truth ──────────────────
        true_col = None
        for candidate in ["RUL_clipped", "RUL", "rul_clipped", "rul"]:
            if candidate in pred_df_cmp.columns:
                true_col = candidate
                break

        if true_col is not None and available:
            last_rows = pred_df_cmp.sort_values(["unit_id", "cycle"]).groupby("unit_id").last().reset_index()
            y_true = last_rows[true_col].values

            rows = []
            for name, col in available.items():
                # Drop NaN predictions instead of filling with 0.
                # NaN = engines with < 30 cycles (LSTM can't form a sequence window).
                # Filling with 0 creates a massive false error vs true RUL ~125 → explodes NASA Score.
                valid_mask = last_rows[col].notna()
                n_missing  = int((~valid_mask).sum())
                y_pred     = last_rows.loc[valid_mask, col].values
                y_true_val = y_true[valid_mask]
                rmse   = float(np.sqrt(np.mean((y_pred - y_true_val) ** 2)))
                mae    = float(np.mean(np.abs(y_pred - y_true_val)))
                ns     = nasa_score(y_true_val, y_pred)
                rows.append({"Model": name, "RMSE": rmse, "MAE": mae,
                             "NASA Score": ns, "_n": int(valid_mask.sum()),
                             "_missing": n_missing})

            metrics_df = pd.DataFrame(rows)

            # ── Metric cards with context ────────────────────────────────────
            st.markdown("#### Performance Metrics (Per-Engine Last Cycle, vs Ground Truth)")
            st.markdown(f"""
            <div class="info-box" style="font-size:13px;">
              Metrics computed on the <b>last recorded cycle per engine</b> (248 test engines), compared against
              ground-truth RUL values from <code>RUL_FD004.txt</code>.<br>
              <b>RMSE</b> = average error in cycles (penalises big mistakes more than MAE) &nbsp;|&nbsp;
              <b>MAE</b> = average absolute error in cycles &nbsp;|&nbsp;
              <b>NASA Score</b> = asymmetric penalty score (lower is better; late predictions cost 3× more than early).<br>
              <b>Note:</b> Engines with fewer than 30 cycles are excluded from LSTM metrics (no sequence window possible).
            </div>""", unsafe_allow_html=True)

            cols_m = st.columns(len(rows))
            best_rmse = min(r["RMSE"] for r in rows)
            for i, (col, row) in enumerate(zip(cols_m, rows)):
                with col:
                    is_best = row["RMSE"] == best_rmse
                    border_color = "#00C853" if is_best else NASA_BLUE
                    crown = " 👑" if is_best else ""
                    missing_note = f"<div class='metric-unit' style='color:#FFA500;'>⚠️ {row['_missing']} engines skipped (< 30 cycles)</div>" if row["_missing"] > 0 else ""
                    st.markdown(f"""
                    <div class="metric-card" style="border-color:{border_color};">
                      <div class="metric-label">{row['Model']}{crown}</div>
                      <div class="metric-value" style="font-size:26px;">RMSE {row['RMSE']:.1f}</div>
                      <div class="metric-unit">MAE {row['MAE']:.1f} cycles</div>
                      <div class="metric-unit">NASA Score {row['NASA Score']:.0f}</div>
                      <div class="metric-unit">{row['_n']} engines evaluated</div>
                      {missing_note}
                    </div>""", unsafe_allow_html=True)

            # ── What do these numbers mean? ──────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 💡 What These Metrics Mean in Practice")
            interp_cols = st.columns(len(rows))
            for col, row in zip(interp_cols, rows):
                with col:
                    rmse = row["RMSE"]; mae = row["MAE"]
                    if rmse < 22:
                        quality = "🟢 **Excellent**"; quality_text = "Errors under ~22 cycles allow confident maintenance scheduling with a small safety buffer."
                    elif rmse < 28:
                        quality = "🟡 **Good**"; quality_text = "Errors in the 22–28 cycle range are acceptable for most maintenance scheduling with a moderate buffer."
                    else:
                        quality = "🟠 **Moderate**"; quality_text = "Errors above 28 cycles require a larger safety buffer but are still useful for trend monitoring."
                    st.markdown(f"""
                    <div class="info-box" style="font-size:12px;">
                      <b>{row['Model']}</b><br>
                      Quality: {quality}<br><br>
                      On average, predictions are off by <b>{mae:.1f} cycles</b> (MAE).
                      95% of predictions fall within ~<b>{rmse*2:.0f} cycles</b> of truth (≈2×RMSE).<br><br>
                      {quality_text}
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Bar chart comparison ──────────────────────────────────────────
            fig_bar = go.Figure()
            colours_m = ["#00B4D8", "#F28E2B", "#00C853"]
            for i, metric in enumerate(["RMSE", "MAE"]):
                fig_bar.add_trace(go.Bar(
                    name=metric, x=metrics_df["Model"], y=metrics_df[metric],
                    marker_color=colours_m[i], text=metrics_df[metric].round(2),
                    textposition="outside", textfont=dict(color=NASA_WHITE)
                ))
            fig_bar.update_layout(barmode="group", title="RMSE & MAE Comparison (lower = better)",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color=NASA_WHITE, legend=dict(font=dict(color=NASA_WHITE)),
                                   height=380, xaxis_title="Model", yaxis_title="Error (cycles)")
            st.plotly_chart(fig_bar, use_container_width=True)

            # NASA Score bar
            fig_ns_live = go.Figure(go.Bar(
                x=metrics_df["Model"], y=metrics_df["NASA Score"],
                marker_color=["#00B4D8", "#F28E2B", "#00C853"][:len(rows)],
                text=metrics_df["NASA Score"].round(0).astype(int),
                textposition="outside", textfont=dict(color=NASA_WHITE)
            ))
            fig_ns_live.add_annotation(text="↓ Lower is better", xref="paper", yref="paper",
                                        x=0.5, y=1.05, showarrow=False, font=dict(color=NASA_GREY))
            fig_ns_live.update_layout(title="NASA Score Comparison (lower = better)",
                                       xaxis_title="Model", yaxis_title="NASA Score",
                                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font_color=NASA_WHITE, height=340)
            st.plotly_chart(fig_ns_live, use_container_width=True)

            # Scatter: predicted vs actual per model
            st.markdown("#### Predicted vs Actual RUL — Per Engine (Last Cycle)")
            scatter_cols = st.columns(len(available))
            for col, (name, pred_col) in zip(scatter_cols, available.items()):
                with col:
                    valid_sc    = last_rows[pred_col].notna()
                    y_p         = last_rows.loc[valid_sc, pred_col].values
                    y_true_sc   = y_true[valid_sc]
                    rmse_sc = float(np.sqrt(np.mean((y_p - y_true_sc)**2)))
                    fig_sc = go.Figure()
                    fig_sc.add_trace(go.Scatter(x=y_true_sc, y=y_p, mode="markers",
                                                 marker=dict(size=5, opacity=0.7, color="#00B4D8"),
                                                 name=name,
                                                 hovertemplate="Actual: %{x:.0f}<br>Pred: %{y:.0f}<extra></extra>"))
                    lim = max(y_true_sc.max(), y_p.max()) + 5
                    fig_sc.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                                 line=dict(color=NASA_RED, dash="dash"), name="Perfect"))
                    fig_sc.update_layout(title=f"{name}<br><sup>RMSE={rmse_sc:.1f} cycles</sup>",
                                          xaxis_title="Actual RUL", yaxis_title="Predicted RUL",
                                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font_color=NASA_WHITE, height=310,
                                          margin=dict(t=55, b=40, l=40, r=20),
                                          showlegend=False)
                    st.plotly_chart(fig_sc, use_container_width=True)

            # Residual histograms
            st.markdown("#### Residual Distributions (Predicted − Actual) — Bias Analysis")
            st.markdown(f"""
            <div class="info-box" style="font-size:13px;">
              Residuals show whether a model is <b>biased</b>. A residual centred at 0 means the model is unbiased.
              <b>Positive mean</b> → model predicts too late on average (overestimates RUL — dangerous in practice).
              <b>Negative mean</b> → model predicts too early (conservative — generates unnecessary maintenance).
            </div>""", unsafe_allow_html=True)
            res_cols = st.columns(len(available))
            res_colours = ["#00B4D8", "#F28E2B", "#00C853"]
            for col, (name, pred_col), colour in zip(res_cols, available.items(), res_colours):
                with col:
                    valid_res = last_rows[pred_col].notna()
                    res = last_rows.loc[valid_res, pred_col].values - y_true[valid_res]
                    bias_dir = "over-predicts ⚠️" if res.mean() > 2 else ("under-predicts" if res.mean() < -2 else "unbiased ✅")
                    fig_r = go.Figure(go.Histogram(x=res, nbinsx=30,
                                                    marker_color=colour, opacity=0.85))
                    fig_r.add_vline(x=0, line_dash="dash", line_color=NASA_RED,
                                    annotation_text="Zero", annotation_font_color=NASA_WHITE)
                    fig_r.add_vline(x=float(res.mean()), line_dash="dot", line_color=NASA_WHITE,
                                    annotation_text=f"μ={res.mean():.1f}", annotation_font_color=NASA_WHITE)
                    fig_r.update_layout(title=f"{name}<br><sup>Mean bias: {res.mean():.1f} ({bias_dir})</sup>",
                                         xaxis_title="Residual (cycles)",
                                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                         font_color=NASA_WHITE, height=300,
                                         margin=dict(t=55, b=40, l=40, r=20))
                    st.plotly_chart(fig_r, use_container_width=True)

        elif not available:
            st.markdown('<div class="error-box">❌ No prediction columns found in test_predictions.csv. '
                        'Expected columns like <code>RF_Predicted_RUL</code>, <code>XGB_Predicted_RUL</code>, '
                        '<code>LSTM_Predicted_RUL</code>.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ No ground-truth RUL column found in test_predictions.csv. '
                        'Expected <code>RUL_clipped</code> or <code>RUL</code>.</div>', unsafe_allow_html=True)

    else:
        # Static comparison table if no predictions file
        st.markdown(f"""
        <div class="info-box">
          Place <code>test_predictions.csv</code> in <code>./lstm/</code> to enable live comparison charts.<br>
          Showing published benchmark results below.
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Published Results — FD004 Test Set")
        cmp_data = {
            "Model":          ["Random Forest (Tuned)", "XGBoost (Tuned)", "CNN-LSTM + Attention"],
            "Approach":       ["Ensemble / Bagging", "Gradient Boosting", "Deep Learning"],
            "Input Format":   ["Flat + rolling stats", "Flat + rolling stats", "3D sequence (30×13)"],
            "RMSE (Test)":    ["~27.6", "~25.1", "~22.7"],
            "MAE (Test)":     ["~17–20", "~16–18", "~14–16"],
            "NASA Score":     ["~34,084", "~8,326", "~3,994"],
            "Training Speed": ["Fast", "Fastest", "Slow (GPU recommended)"],
        }
        st.dataframe(pd.DataFrame(cmp_data), use_container_width=True)

        st.markdown("#### 💡 What These Metrics Mean")
        st.markdown(f"""
        <div class="info-box">
          <b>RMSE (Root Mean Squared Error):</b> Average prediction error in engine cycles, with heavier 
          penalty for large misses. An RMSE of ~22.7 means the LSTM is typically off by ~23 cycles — 
          with a 125-cycle RUL cap, that's roughly <b>18% average error</b>.<br><br>
          <b>MAE (Mean Absolute Error):</b> Average absolute error in cycles. Lower is better; smaller MAE
          means predictions are closer to true RUL on average — enabling tighter maintenance scheduling.<br><br>
          <b>NASA Score:</b> The official asymmetric metric where late predictions (model says engine is healthy 
          when it's actually failing) cost <b>3× more</b> than early predictions. Lower is strictly better.
          The LSTM's score of ~3,994 vs RF's ~34,084 shows it makes far fewer dangerously late predictions.<br><br>
          <b>Bottom line:</b> The CNN-LSTM achieves the best results across all three metrics. Random Forest 
          and XGBoost are still viable, especially when interpretability or training speed matters more than 
          peak accuracy.
        </div>""", unsafe_allow_html=True)

        # NASA score bar chart (static) — values match sidebar reference metrics
        fig_ns = go.Figure(go.Bar(
            x=["Random Forest", "XGBoost", "CNN-LSTM"],
            y=[34084, 8326, 3994],
            marker_color=[NASA_GREY, "#00B4D8", "#00C853"],
            text=["~34,084", "~8,326", "~3,994"], textposition="outside",
            textfont=dict(color=NASA_WHITE)
        ))
        fig_ns.add_annotation(text="↓ Lower is better", xref="paper", yref="paper",
                              x=0.5, y=1.05, showarrow=False, font=dict(color=NASA_GREY))
        fig_ns.update_layout(title="NASA Score Comparison (lower = better)",
                              xaxis_title="Model", yaxis_title="NASA Score",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color=NASA_WHITE, height=380)
        st.plotly_chart(fig_ns, use_container_width=True)

    # Model strengths table (always shown)
    st.markdown("#### Model Trade-off Summary")
    trade_df = pd.DataFrame({
        "Model":            ["Random Forest", "XGBoost", "CNN-LSTM", "K-Means"],
        "Type":             ["Supervised", "Supervised", "Supervised", "Unsupervised"],
        "Key Strength":     ["Robust, interpretable, OOB", "Fast, regularised, SHAP", "Best accuracy, temporal", "No labels needed"],
        "Key Weakness":     ["Slower training", "More hyperparams", "Needs GPU, complex", "No RUL output"],
        "Input Format":     ["Flat + rolling", "Flat + rolling", "30×13 sequence", "Mean of last 30"],
        "Interpretability": ["High (feature importance)", "High (SHAP)", "Medium (attention)", "High (cluster profiles)"],
    })
    st.dataframe(trade_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### 📖 About This Application")

    # ── How the sample scenarios work ────────────────────────────────────────
    with st.expander("🔬 How Sample Scenarios Work — and Why Models Predict Different RULs", expanded=True):
        st.markdown(f"""
<div style="color:{NASA_LIGHT};line-height:1.7;font-size:13.5px;">

<b style="font-size:15px;">Where the scenarios come from</b><br>
The three sample engine scenarios (Healthy, Moderate, Critical) are <b>not fabricated</b>.
Every sensor value — starting level, direction of drift, and magnitude — was extracted
directly from real FD004 training engines:

<ul>
  <li><b>Healthy:</b> first-45-row windows of 249 training engines where RUL_clipped = 125 (brand-new life phase).
      Per-sensor mean and drift were averaged across all 249 engines.</li>
  <li><b>Moderate:</b> 45-row windows centred at the row where each engine's RUL_clipped ≈ 60 (mid-life).
      Same averaging procedure.</li>
  <li><b>Critical:</b> last-45-row windows of engines with RUL_clipped ≤ 20 (approaching failure).
      Start values = real_rmean − drift/2 so the 30-cycle rolling mean of the window hits
      the real critical-zone mean exactly.</li>
</ul>

Noise amplitude = 0.9 std — the same unit variance produced by StandardScaler, which is what
any real 30-row window has after per-cluster normalisation. This gives rolling std ≈ 0.90
in the templates, matching the 0.85–0.94 range seen in actual training rows.

<br><br>
<b style="font-size:15px;">Why RF and XGBoost give different numbers for the same file</b><br>
Both RF and XGBoost receive <b>the same 37-feature vector</b> after the app's rolling pipeline:
12 raw sensors + 12 rolling means + 12 rolling stds + op_condition.
They produce different RUL values because they learned different decision boundaries
during training — RF averages across 300 trees each making axis-aligned splits;
XGBoost builds an additive sequence of shallow trees using gradient boosting.
The key discriminating features are <code>sensor_14_rmean</code>, <code>sensor_04_rmean</code>,
and <code>sensor_03_rmean</code>. For the critical scenario, real training data has:
<code>sensor_14_rmean ≈ 0.40</code>, <code>sensor_04_rmean ≈ 0.22</code>. Our template produces
~0.65 and ~0.35 — slightly above real (because a 45-row window cannot perfectly replicate
a full-lifecycle accumulation) but well within the critical zone.
For the critical scenario specifically, <b>RF and XGBoost receive the identical CSV</b>
(shared seed), so any RUL gap between them is purely a model boundary difference.

<br><br>
<b style="font-size:15px;">Why LSTM gives very different values from RF/XGB scenarios</b><br>
LSTM uses a <b>different 12-sensor set</b> (sensor_02, 03, 04, 07, 08, 09, 11, 12, 13, 14, 15, 17 —
a non-contiguous subset; sensor_11 and sensor_15 are LSTM-only and absent from RF/XGB files) and a completely different input format (no unit_id, no cycle).
If you upload an RF template to the LSTM tab, the column mapping will be wrong — the LSTM reads
numeric columns in order and the column assignments will not match what it expects.
Each model tab's scenarios are generated specifically for that model. This is by design:
the three models were trained on overlapping but distinct feature representations.

<br><br>
<b style="font-size:15px;">The key FD004 degradation signal</b><br>
FD004 has two simultaneous fault modes: HPC (High Pressure Compressor) degradation and
fan degradation. The dominant sensor signals are:
<ul>
  <li><b>sensor_14 (HPC outlet temperature):</b> rises steeply near failure — the single
      most important feature. Rolling mean goes from −0.15 (healthy) → +0.02 (moderate) → +0.40 (critical).</li>
  <li><b>sensor_04 (total pressure ratio):</b> rises with compressor wear. −0.10 → +0.04 → +0.22.</li>
  <li><b>sensor_03, 09, 17:</b> secondary HPC indicators, all trending upward with degradation.</li>
  <li><b>sensor_13, 08 (bypass ratios):</b> drop with fan degradation — negative drift is the signal.</li>
</ul>
RF and XGBoost learned splits on these rolling mean values.
The LSTM learned the temporal pattern in the raw sequence — it sees 30 consecutive cycles
and detects the rate of change, not just the current level.
This is why LSTM can sometimes give a different RUL than RF/XGB even on equivalent
health-state data: it weighs the <i>trajectory</i> rather than the <i>snapshot</i>.

</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        **NASA C-MAPSS FD004 Dataset**

        The Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)
        simulates turbofan engine degradation. FD004 is the most challenging subset:

        - 249 training engines / 248 test engines
        - 6 distinct operating conditions (altitude × Mach)
        - 2 simultaneous fault modes (HPC + fan degradation)
        - 21 raw sensors → 12 selected after preprocessing

        **Preprocessing pipeline:**
        1. Drop constant / near-constant sensors (std < 10)
        2. Compute RUL → clip at 125 cycles (piecewise linear target)
        3. Rolling-mean smoothing (window = 5 cycles, per engine)
        4. K-Means clustering of operating conditions (k=6)
        5. Per-cluster StandardScaler normalisation (fit on train only)
        6. Sliding-window sequences (length = 30) for LSTM
        7. 30-cycle rolling stats (mean + std) for RF/XGBoost
        """)

    with c2:
        st.markdown(f"""
        **LSTM Architecture (CNN → LSTM → Bahdanau Attention)**

        ```
        Input  (30 × 13)
            │
        Conv1D(48, k=3, causal) → ReLU
        Conv1D(48, k=3, causal) → ReLU
        Dropout(0.1)
            │
        LSTM(96, return_sequences=True)
            │
        Bahdanau Attention
          score(hₜ) = v · tanh(W · hₜ)
          αₜ        = softmax(scores)
          context   = Σ αₜ · hₜ
            │
        Dropout(0.2) → BatchNorm
        Dense(48, relu) → Dense(1, linear)
            │
        RUL output (cycles)
        ```

        **Loss: Failure-Region Weighted Huber**
        - Huber(δ=15) as base
        - 3× weight when true RUL < 50 cycles
        """)

    st.markdown("---")
    st.markdown("#### 📐 Evaluation Metrics")
    metrics_info = pd.DataFrame({
        "Metric":      ["RMSE", "MAE", "NASA Score"],
        "Formula":     ["√( Σ(ŷ−y)² / n )", "Σ|ŷ−y| / n",
                        "Σ exp(d/10)−1 if d≥0, else exp(−d/13)−1"],
        "Description": ["Penalises large errors more heavily",
                        "Average error in cycles — easier to interpret",
                        "Asymmetric: late predictions penalised 3× more than early ones"],
        "Units":       ["cycles", "cycles", "dimensionless (lower is better)"],
    })
    st.dataframe(metrics_info, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📂 Required Artifact Files in `./lstm/`")
    files_info = pd.DataFrame({
        "File": [
            "lstm_fd004_model.keras",
            "lstm_fd004_preprocessors.pkl",
            "lstm_fd004_config.pkl",
            "rf_best_model.pkl",
            "xgb_best_model.pkl",
            "kmeans_model.pkl",
            "op_condition_kmeans.pkl",
            "mm_scaler_cluster.pkl",
            "test_predictions.csv",
            "train_regression.csv",
        ],
        "Used by": [
            "LSTM Tab", "LSTM Tab", "LSTM Tab",
            "Random Forest Tab", "XGBoost Tab",
            "K-Means Tab (optional)", "K-Means Tab (optional)", "K-Means Tab (optional)",
            "EDA Tab + Model Comparison Tab",
            "EDA Tab + K-Means Explorer",
        ],
        "Generated in": [
            "Notebook Part 4", "Notebook Part 4", "Notebook Part 4",
            "Notebook Part 2", "Notebook Part 3",
            "Notebook Part 5", "Notebook Part 1", "Notebook Part 1",
            "Notebook Part 7", "Notebook Part 7",
        ],
    })
    st.dataframe(files_info, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"""
    <div class="info-box">
      <b>Team:</b> Ali Ahsan · Anas Bin Waheed · Haqiq Azeem Khan<br>
      <b>Course:</b> Machine Learning — NUST<br>
      <b>Dataset:</b> NASA C-MAPSS FD004 (Turbofan Engine Degradation Simulation)<br>
      <b>App version:</b> 2.1 — K-Means Explorer · CSV Templates · Enhanced EDA
    </div>
    """, unsafe_allow_html=True)
