# 🚀 NASA C-MAPSS FD004 — Turbofan Engine RUL Predictor

A production-ready Streamlit application for predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA C-MAPSS FD004 dataset. Three trained models — a CNN-LSTM with Bahdanau attention, a tuned Random Forest, and a tuned XGBoost — run live in the browser. An interactive K-Means explorer, full EDA dashboard, and head-to-head model comparison are also included.

---

## 📋 Table of Contents

- [Live Demo](#live-demo)
- [Application Overview](#application-overview)
- [Project Structure](#project-structure)
- [Required Artifacts](#required-artifacts)
- [Local Setup](#local-setup)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [GitHub Setup](#github-setup)
- [How Each Model Works](#how-each-model-works)
- [Data & Preprocessing Summary](#data--preprocessing-summary)
- [Troubleshooting](#troubleshooting)

---

## 🌐 Live Demo

> Deploy your own copy via the instructions below, or share the Streamlit Cloud link after deployment.

---

## 🖥 Application Overview

The app is organised into **7 tabs**:

| Tab | Description |
|---|---|
| ⚡ **LSTM** | CNN-LSTM + Bahdanau Attention — upload 30+ rows or use built-in scenarios |
| 🌲 **Random Forest** | Tuned RF Regressor — flat 37-feature input, rolling statistics included |
| 🔥 **XGBoost** | Tuned XGBoost Regressor — same feature format as RF |
| 🔵 **K-Means Explorer** | Cluster health states (K=2–5), elbow/silhouette analysis, PCA plots |
| 📊 **EDA** | Dataset explorer — per-engine degradation trends, sensor heatmaps, RUL distributions |
| 📈 **Model Comparison** | Head-to-head RMSE / MAE / NASA Score using `test_predictions.csv` |
| 📖 **About** | Architecture diagrams, preprocessing pipeline, methodology notes |

Each prediction tab has:
- **Try Sample Engine** — three built-in scenarios (Healthy / Moderate / Critical) generated from real FD004 training data distributions
- **Upload your own CSV** — live inference with validation warnings
- **Status badge** — colour-coded Critical / Warning / Healthy classification
- **Gauge + heatmap** — visual output for every prediction

---

## 📁 Project Structure

```
rul-predictor/
│
├── app.py                        ← Main Streamlit application
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
├── .streamlit/
│   └── config.toml               ← Optional: theme / server config
│
└── (model artifacts — all in the root folder next to app.py)
    ├── lstm_fd004_model.keras    ← Trained CNN-LSTM weights
    ├── lstm_fd004_preprocessors.pkl  ← StandardScalers per op-condition + op_feat_idx
    ├── lstm_fd004_config.pkl     ← {SEQ_LEN, N_FEAT, RUL_MAX}
    ├── rf_best_model.pkl         ← Trained Random Forest pipeline
    ├── xgb_best_model.pkl        ← Trained XGBoost pipeline
    ├── kmeans_model.pkl          ← K-Means model (best K from training)
    ├── op_condition_kmeans.pkl   ← K-Means used for operating-condition discovery
    ├── mm_scaler_cluster.pkl     ← MinMaxScaler for cluster feature normalisation
    ├── test_predictions.csv      ← Pre-computed per-engine predictions (Model Comparison tab)
    └── train_regression.csv      ← Processed training data (EDA tab)
```

> **All ten artifact files must sit in the same directory as `app.py`.**  
> The app uses `ASSET_DIR = "."` — there is no `lstm/` subfolder.

---

## 📦 Required Artifacts

| File | Size (approx.) | Purpose |
|---|---|---|
| `lstm_fd004_model.keras` | ~5–20 MB | LSTM weights & architecture |
| `lstm_fd004_preprocessors.pkl` | < 1 MB | Per-condition StandardScaler + op_feat_idx |
| `lstm_fd004_config.pkl` | < 1 KB | SEQ_LEN=30, N_FEAT=13, RUL_MAX=125 |
| `rf_best_model.pkl` | ~50–200 MB | Random Forest (sklearn Pipeline) |
| `xgb_best_model.pkl` | ~5–30 MB | XGBoost Booster |
| `kmeans_model.pkl` | < 1 MB | K-Means for health clustering |
| `op_condition_kmeans.pkl` | < 1 MB | K-Means for operating-condition labels |
| `mm_scaler_cluster.pkl` | < 1 MB | MinMaxScaler for clustering |
| `test_predictions.csv` | ~1–5 MB | Per-engine predictions for all models |
| `train_regression.csv` | ~5–20 MB | Processed training data |

The app runs **without any artifact** — missing files degrade gracefully (prediction tabs show a warning; sample scenario panels still work). The more artifacts are present, the more features unlock.

---

## 💻 Local Setup

### 1. Prerequisites

- Python **3.10** or **3.11** (TensorFlow 2.15+ requirement)
- `pip` ≥ 23.0
- Git

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/rul-predictor.git
cd rul-predictor
```

### 3. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **Apple Silicon (M1/M2/M3):** Use `tensorflow-macos` and `tensorflow-metal` instead of `tensorflow`. Replace the tensorflow line in `requirements.txt` with:
> ```
> tensorflow-macos>=2.15.0
> tensorflow-metal>=1.1.0
> ```

### 5. Place model artifacts

Copy all ten artifact files into the **project root** (same folder as `app.py`):

```bash
# Example — adjust source path to wherever your artifacts are:
cp /path/to/your/artifacts/* .
```

### 6. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## ☁️ Streamlit Cloud Deployment

### Step 1 — Push to GitHub

Follow the [GitHub Setup](#github-setup) section below first.

### Step 2 — Sign in to Streamlit Cloud

Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.

### Step 3 — New app

Click **"New app"** → select your repository → set:

| Field | Value |
|---|---|
| Repository | `<your-username>/rul-predictor` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version | `3.11` |

### Step 4 — Upload model artifacts (Secrets / File uploader)

Streamlit Cloud does **not** allow uploading binary files via the UI directly. Use one of these approaches:

#### Option A — Git LFS (recommended for files < 100 MB each)

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.keras" "*.pkl"
git add .gitattributes
git add *.keras *.pkl *.csv
git commit -m "Add model artifacts via LFS"
git push origin main
```

Then re-deploy on Streamlit Cloud — it will pull LFS files automatically.

#### Option B — Download from URL at startup

Add a `startup.py` or put download logic in `app.py` that fetches from Google Drive / Hugging Face Hub / S3 on first run.

Example using Hugging Face Hub:

```python
# In app.py, before loading models:
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="<your-hf-repo>", filename="lstm_fd004_model.keras", local_dir=".")
```

Then add `huggingface_hub` to `requirements.txt`.

#### Option C — Include small artifacts directly in the repo

Files under ~50 MB can be committed without LFS (GitHub's soft limit is 50 MB per file, hard limit 100 MB). `*.pkl` scalers and configs are usually small enough.

### Step 5 — Deploy

Click **"Deploy"**. Streamlit Cloud installs `requirements.txt` and starts the app. First cold start may take 2–5 minutes.

### Step 6 — Optional: Custom theme

Create `.streamlit/config.toml` in your repo:

```toml
[theme]
primaryColor = "#0B3D91"
backgroundColor = "#0a0f1e"
secondaryBackgroundColor = "#0d1b2a"
textColor = "#FFFFFF"
font = "sans serif"

[server]
maxUploadSize = 50
```

---

## 🐙 GitHub Setup

### First-time push

```bash
# Initialise repo (skip if already a git repo)
git init
git branch -M main

# Add all files
git add app.py requirements.txt README.md .streamlit/

# Add artifacts (large files via LFS — see above)
git add *.keras *.pkl *.csv

git commit -m "Initial commit — NASA RUL Predictor"

# Add remote and push
git remote add origin https://github.com/<your-username>/rul-predictor.git
git push -u origin main
```

### Updating after changes

```bash
git add app.py
git commit -m "Fix: correct sensor list in LSTM template description"
git push
```

Streamlit Cloud auto-redeploys on every push to `main`.

### Recommended `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
venv/
.env

# Streamlit cache
.streamlit/secrets.toml

# Large model files (if NOT using Git LFS)
# Uncomment these if you're hosting artifacts elsewhere:
# *.keras
# *.pkl
# train_regression.csv
# test_predictions.csv

# OS
.DS_Store
Thumbs.db
```

---

## 🧠 How Each Model Works

### CNN-LSTM + Bahdanau Attention

**Input:** Sliding window of 30 consecutive engine cycles × 13 features (12 normalised sensors + operating condition)

**Architecture:**
```
Input (30 × 13)
  → Conv1D (64 filters, kernel=3, ReLU)   — local pattern extraction
  → Conv1D (128 filters, kernel=3, ReLU)
  → MaxPooling1D (pool=2)
  → Bidirectional LSTM (128 units)         — temporal dependencies
  → Bahdanau Attention                     — weighted focus on critical timesteps
  → Dense (64, ReLU) → Dropout(0.3)
  → Dense (1, Linear)                      — RUL output, clipped [0, 125]
```

**LSTM sensors:** `sensor_02, 03, 04, 07, 08, 09, 11, 12, 13, 14, 15, 17`
**Performance (248 test engines):** RMSE ≈ 22.7 · NASA Score ≈ 3,994 (237 engines with valid predictions)

---

### Random Forest Regressor

**Input:** Flat 37-feature vector (last row of the rolling-feature-augmented engine history)

**Features:** 12 raw sensor values + 12 rolling 30-cycle means + 12 rolling 30-cycle standard deviations + 1 operating condition = 37 total

**RF sensors:** `sensor_01, 02, 03, 04, 07, 08, 09, 12, 13, 14, 17, 18`

**Performance:** RMSE ≈ 27.6 · NASA Score ≈ 34,084

---

### XGBoost Regressor

**Input:** Same 37-feature flat vector as Random Forest — the two models are directly comparable on identical CSVs.

**Performance:** RMSE ≈ 25.1 · NASA Score ≈ 8,326

---

### K-Means Clustering

**Purpose:** Unsupervised discovery of engine health states. Not used for RUL regression — used to label operating conditions and visualise degradation clusters.

**K range:** 2–5 (best K selected by silhouette score)

---

## 🔬 Data & Preprocessing Summary

| Step | Detail |
|---|---|
| Raw dataset | NASA C-MAPSS FD004 — 249 training / 248 test engines |
| Operating conditions | 6 real conditions; 2 simultaneous fault modes (HPC + fan degradation) |
| Sensor drop | Removed 9 near-constant sensors (std < 10): 05, 06, 10, 11, 15, 16, 19, 20, 21 + op_set_3 |
| Surviving sensors | 12 (sensor_01, 02, 03, 04, 07, 08, 09, 12, 13, 14, 17, 18) |
| Op-condition discovery | K-Means (K=6) on [op_set_1, op_set_2] → condition labels 0–5 |
| Normalisation | StandardScaler fitted per condition on training data only |
| RUL target | Piece-wise linear, clipped at 125 cycles |
| LSTM window | 30 cycles, stride 1 → shape (N, 30, 13) |
| RF/XGB rolling stats | 30-cycle rolling mean + std per sensor, `fillna(0)` on std only |

---

## 🛠 Troubleshooting

| Problem | Solution |
|---|---|
| `File not found: lstm_fd004_model.keras` | Place all artifact `.pkl`/`.keras`/`.csv` files in the same directory as `app.py` |
| `ModuleNotFoundError: tensorflow` | Run `pip install tensorflow>=2.15.0` or use `tensorflow-macos` on Apple Silicon |
| App loads but prediction returns NaN | Check that `lstm_fd004_preprocessors.pkl` was trained with the same sensor order as the CSV |
| Streamlit Cloud: no model files | Use Git LFS or add a startup download script (see Deployment section) |
| `KeyError: 'op_feat_idx'` | Regenerate `lstm_fd004_preprocessors.pkl` — older version missing this key |
| `XGBoost version mismatch` | Match xgboost version between training environment and `requirements.txt` |
| Upload warning: "Need at least 30 rows" | Your CSV has fewer than 30 rows — LSTM needs a full window; RF/XGB also warn below 30 |
| Comparison tab empty | Place `test_predictions.csv` next to `app.py` — it must have columns: `unit_id, cycle, true_rul, lstm_pred, rf_pred, xgb_pred` |

---

## 📄 License

This project uses the publicly available [NASA C-MAPSS dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). The application code is released under the MIT License.

---

## 🙏 Acknowledgements

- NASA Prognostics Center of Excellence — C-MAPSS dataset
- Saxena et al. (2008) — original CMAPSS benchmark paper
- Streamlit, TensorFlow/Keras, scikit-learn, XGBoost, Plotly
