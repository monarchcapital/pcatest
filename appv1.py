# -*- coding: utf-8 -*-
# Brazil DI Futures PCA — stable forecast + robust in-sample reconstruction
#
# ENHANCEMENTS:
#  - Streamlined UI to focus on core yield curve visualizations.
#  - Added VAR (Vector Autoregression) model for forecasting.
#  - Added ARIMA (Autoregressive Integrated Moving Average) model for forecasting.
#  - User can select the forecasting model from the sidebar.
#  - Improved visualizations for clarity and focus on outright yield curve values.
#  - **REVISED: Output is now mapped back to actual contract tickers.**
#  - **NEW: Added a default "PCA Fair Value" forecast model.**
#  - **NEW: Added a "Fair Value Spread Analysis" section to identify mispricing.**
#  - **NEW: Added a PCA components heatmap for intuitive analysis.**
#
# Includes:
#  - No "fallback" bug: reconstruction honors the chosen date exactly.
#  - Clean separation of in-sample reconstruction vs next-BD forecast.
#  - Grid rule: 0–3y quarterly (0.25y), 3–5y half-year (0.5y), >5y yearly (1.0y).
#  - Compounding choice (identity or DI daily-compounding → annualized zero).
#
# -------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings from statsmodels
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Brazil DI PCA — Forecast & Tools")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# --------------------------
# Helpers
# --------------------------
def safe_to_datetime(s):
    # Explicitly state that the day is NOT first to resolve ambiguity for dates like 08-12-2025
    return pd.to_datetime(s, errors='coerce', dayfirst=False)

def normalize_rate_input(val, unit):
    """Return rate in decimal fraction (e.g., 0.1345 for 13.45%)."""
    if pd.isna(val):
        return np.nan
    v = float(val)
    if "Percent" in unit:
        return v / 100.0
    if "Basis" in unit:
        return v / 10000.0
    return v  # decimal

def denormalize_to_percent(frac):
    if pd.isna(frac):
        return np.nan
    return 100.0 * float(frac)

def np_busdays_exclusive(start_dt, end_dt, holidays_np):
    if pd.isna(start_dt) or pd.isna(end_dt):
        return 0
    s = np.datetime64(pd.Timestamp(start_dt).date()) + np.timedelta64(1, "D")
    e = np.datetime64(pd.Timestamp(end_dt).date())
    if e < s:
        return 0
    return int(np.busday_count(s, e, weekmask="1111100", holidays=holidays_np))

def calculate_ttm(valuation_ts, expiry_ts, holidays_np, year_basis):
    bd = np_busdays_exclusive(valuation_ts, expiry_ts, holidays_np)
    return np.nan if bd <= 0 else bd / float(year_basis)

def next_business_day(date_like, holidays_np):
    d = np.datetime64(pd.Timestamp(date_like).date())
    nxt = np.busday_offset(d, 1, weekmask="1111100", holidays=holidays_np)
    return pd.Timestamp(nxt)

def build_std_grid_by_rule(max_year=7.0):
    a = list(np.round(np.arange(0.25, 3.0 + 0.001, 0.25), 2))   # 0–3y quarterly
    b = list(np.round(np.arange(3.5, 5.0 + 0.001, 0.5), 2))     # 3–5y half-year
    c = list(np.round(np.arange(6.0, max_year + 0.001, 1.0), 2))# >5y yearly
    return a + b + c

# --------------------------
# Sidebar — Inputs
# --------------------------
st.sidebar.header("Upload / Settings")

yield_file = st.sidebar.file_uploader("1) Yield data CSV (dates + contract columns)", type="csv", key="yield_file")
expiry_file = st.sidebar.file_uploader("2) Expiry mapping CSV (maturity, date)", type="csv", key="expiry_file")
holiday_file = st.sidebar.file_uploader("3) Holiday dates CSV (optional)", type="csv", key="holiday_file")

interp_method = st.sidebar.selectbox("Interpolation method", ["linear", "cubic", "quadratic", "nearest"])
apply_smoothing = st.sidebar.checkbox("Apply smoothing (3-day centered)", value=False)
n_components_sel = st.sidebar.slider("Number of PCA components", 1, 12, 3)

rate_unit = st.sidebar.selectbox(
    "Input rate unit",
    ["Percent (e.g. 13.45)", "Decimal (e.g. 0.1345)", "Basis points (e.g. 1345)"],
)
year_basis = int(st.sidebar.selectbox("Business days in year", [252, 360], index=0))

compounding_model = st.sidebar.radio(
    "Compounding model for curve construction",
    ["identity (input already annual zero / effective)", "DI daily business compounding"],
    index=0,
)

use_grid_rule = st.sidebar.checkbox(
    "Use standard grid rule",
    value=True,
)

std_maturities_txt = st.sidebar.text_input(
    "Or custom standard maturities (years, comma-separated)",
    "0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.50,4.00,4.50,5.00,6.00,7.00",
)

st.sidebar.markdown("---")
st.sidebar.header("Forecasting Model")

forecast_model_type = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["None (PCA Fair Value)", "Average Delta (Momentum)", "VAR (Vector Autoregression)", "ARIMA (per Component)"]
)

# --- Model-specific parameters ---
if forecast_model_type == "Average Delta (Momentum)":
    rolling_window_days = st.sidebar.number_input("PCs avg-delta window (days)", min_value=1, max_value=20, value=5, step=1)
    pc_damp = st.sidebar.slider("Damping for PC2..PCn (0 = ignore, 1 = full)", 0.0, 1.0, 0.5, 0.05)
elif forecast_model_type == "VAR (Vector Autoregression)":
    var_lags = st.sidebar.number_input("VAR model lags", min_value=1, max_value=20, value=1, step=1)
elif forecast_model_type == "ARIMA (per Component)":
    st.sidebar.write("Using ARIMA(1,1,0) for each PC.")

apply_cap = st.sidebar.checkbox("Apply daily cap to predicted curve change", value=True)
cap_bps = st.sidebar.number_input("Cap per-day move (bps, ±)", min_value=0, max_value=500, value=10, step=1)
cap_frac = float(cap_bps) / 10000.0 # Convert bps to decimal for calculation

st.sidebar.markdown("---")
show_previous_curve = st.sidebar.checkbox("Show Previous Day's Curve", value=True)


if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

@st.cache_data(show_spinner=False)
def _quick_read_csv(upload):
    return pd.read_csv(io.StringIO(upload.getvalue().decode("utf-8")))

yields_df_head = None
if yield_file is not None:
    try:
        tmp = _quick_read_csv(yield_file)
        if not tmp.empty:
            dc = tmp.columns[0]
            tmp[dc] = safe_to_datetime(tmp[dc])
            tmp = tmp.dropna(subset=[dc]).set_index(dc)
            yields_df_head = tmp
    except Exception:
        yields_df_head = None

if yields_df_head is not None and not yields_df_head.empty:
    min_date_available = yields_df_head.index.min().date()
    max_date_available = yields_df_head.index.max().date()
else:
    min_date_available = pd.to_datetime("2000-01-01").date()
    max_date_available = pd.to_datetime("2100-01-01").date()

date_train_start = st.sidebar.date_input("PCA Training Start Date", value=min_date_available, min_value=min_date_available, max_value=max_date_available)
date_train_end = st.sidebar.date_input("PCA Training End Date", value=max_date_available, min_value=min_date_available, max_value=max_date_available)
if date_train_start > date_train_end:
    st.sidebar.error("PCA Training Start Date cannot be after End Date.")

inputs_ready = (yield_file is not None and expiry_file is not None and (use_grid_rule or len(std_maturities_txt.strip()) > 0) and date_train_start <= date_train_end)

col_a, col_b = st.sidebar.columns(2)
if not st.session_state.analysis_started:
    if inputs_ready and col_a.button("Start Analysis"):
        st.session_state.analysis_started = True
        st.rerun()
else:
    if col_b.button("Stop Analysis"):
        st.session_state.analysis_started = False
        st.stop()

if not st.session_state.analysis_started:
    st.info("Upload files, set parameters and training date range, then click **Start Analysis**.")
    st.stop()

# --------------------------
# Load full data
# --------------------------
def load_csv_file(f):
    return pd.read_csv(io.StringIO(f.getvalue().decode("utf-8")))

yields_df = load_csv_file(yield_file)
date_col = yields_df.columns[0]
yields_df[date_col] = safe_to_datetime(yields_df[date_col])
yields_df = yields_df.dropna(subset=[date_col]).set_index(date_col).sort_index()
yields_df.columns = [str(c).strip() for c in yields_df.columns]
for c in yields_df.columns:
    yields_df[c] = pd.to_numeric(yields_df[c], errors="coerce")

expiry_raw = load_csv_file(expiry_file)
expiry_df = expiry_raw.iloc[:, :2].copy()
expiry_df.columns = ["MATURITY", "DATE"]
expiry_df["MATURITY"] = expiry_df["MATURITY"].astype(str).str.strip().str.upper()
expiry_df["DATE"] = safe_to_datetime(expiry_df["DATE"])
expiry_df = expiry_df.dropna(subset=["DATE"]).set_index("MATURITY")

holidays_np = np.array([], dtype="datetime64[D]")
if holiday_file:
    hol_df = load_csv_file(holiday_file)
    hol_series = safe_to_datetime(hol_df.iloc[:, 0]).dropna()
    if not hol_series.empty:
        holidays_np = np.array(hol_series.dt.date, dtype="datetime64[D]")

# Filter for training window (used for PCA)
train_mask = (yields_df.index.date >= date_train_start) & (yields_df.index.date <= date_train_end)
yields_df_train = yields_df.loc[train_mask].sort_index()
if yields_df_train.empty:
    st.error("No yields in the selected PCA training date range.")
    st.stop()
if apply_smoothing:
    yields_df_train = yields_df_train.rolling(window=3, min_periods=1, center=True).mean()

# Standard maturities grid
if use_grid_rule:
    std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)
else:
    try:
        std_maturities = [float(x.strip()) for x in std_maturities_txt.split(",") if x.strip() != ""]
        std_arr = np.array(sorted(std_maturities), dtype=float)
    except Exception:
        st.error("Error parsing standard maturities.")
        st.stop()
std_cols = [f"{m:.2f}Y" for m in std_arr]

# --------------------------
# Core Data Transformation Functions
# --------------------------
def row_to_std_grid(dt, row_series, available_contracts, expiry_df, std_arr, holidays_np, year_basis, rate_unit, compounding_model, interp_method):
    ttm_list, zero_list = [], []
    for col in available_contracts:
        mat_up = str(col).strip().upper()
        if mat_up not in expiry_df.index:
            continue
        exp = expiry_df.loc[mat_up, "DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date() < dt.date():
            continue
        t = calculate_ttm(dt, exp, holidays_np, year_basis)
        if np.isnan(t) or t <= 0:
            continue
        raw_val = row_series.get(col, np.nan)
        if pd.isna(raw_val):
            continue
        r_frac = normalize_rate_input(raw_val, rate_unit)
        if compounding_model.startswith("identity"):
            zero_frac = r_frac
        else:
            T = int(np_busdays_exclusive(dt, exp, holidays_np))
            if T == 0: continue
            r_daily = r_frac / float(year_basis)
            DF = (1.0 + r_daily) ** (-T)
            zero_frac = DF ** (-1.0 / t) - 1.0
        ttm_list.append(t)
        zero_list.append(denormalize_to_percent(zero_frac))
    if len(ttm_list) > 1 and len(set(np.round(ttm_list, 12))) > 1:
        try:
            order = np.argsort(ttm_list)
            f = interp1d(
                np.array(ttm_list)[order],
                np.array(zero_list)[order],
                kind=interp_method,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            return f(std_arr)
        except Exception:
            return np.full_like(std_arr, np.nan, dtype=float)
    return np.full_like(std_arr, np.nan, dtype=float)

def std_grid_to_contracts(dt, std_curve_rates, std_arr, expiry_df, all_contracts, holidays_np, year_basis, rate_unit, compounding_model, interp_method):
    """Converts a curve on the standard grid back to individual contract rates."""
    contract_rates = pd.Series(index=all_contracts, dtype=float)
    
    interp_func = interp1d(std_arr, std_curve_rates, kind=interp_method, bounds_error=False, fill_value='extrapolate')

    for contract in all_contracts:
        mat_up = str(contract).strip().upper()
        if mat_up not in expiry_df.index:
            continue
        
        exp = expiry_df.loc[mat_up, "DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date() < dt.date():
            continue

        t = calculate_ttm(dt, exp, holidays_np, year_basis)
        if np.isnan(t) or t <= 0:
            continue

        zero_percent = interp_func(t)
        zero_frac = zero_percent / 100.0

        if compounding_model.startswith("identity"):
            rate_frac = zero_frac
        else:
            T = int(np_busdays_exclusive(dt, exp, holidays_np))
            if T == 0: continue
            DF = (1.0 + zero_frac) ** (-t)
            r_daily = DF ** (-1.0 / T) - 1.0
            rate_frac = r_daily * float(year_basis)
        
        if "Percent" in rate_unit:
             contract_rates[contract] = rate_frac * 100.0
        elif "Basis" in rate_unit:
             contract_rates[contract] = rate_frac * 10000.0
        else:
             contract_rates[contract] = rate_frac

    return contract_rates.dropna()


@st.cache_data
def build_pca_matrix(yields_df_train, expiry_df, std_arr, holidays_np, year_basis, rate_unit, compounding_model, interp_method):
    pca_df = pd.DataFrame(np.nan, index=yields_df_train.index, columns=std_cols, dtype=float)
    available_contracts = yields_df_train.columns

    for dt in yields_df_train.index:
        pca_df.loc[dt] = row_to_std_grid(
            dt, yields_df_train.loc[dt], available_contracts, expiry_df, 
            std_arr, holidays_np, year_basis, rate_unit, compounding_model, interp_method
        )

    pca_vals = pca_df.values.astype(float)
    col_means = np.nanmean(pca_vals, axis=0)
    if np.isnan(col_means).any():
        overall_mean = np.nanmean(col_means[~np.isnan(col_means)]) if np.any(~np.isnan(col_means)) else 0.0
        col_means = np.where(np.isnan(col_means), overall_mean, col_means)
    inds = np.where(np.isnan(pca_vals))
    if inds[0].size > 0:
        pca_vals[inds] = np.take(col_means, inds[1])
    return pd.DataFrame(pca_vals, index=pca_df.index, columns=pca_df.columns)


pca_df_filled = build_pca_matrix(yields_df_train, expiry_df, std_arr, holidays_np, year_basis, rate_unit, compounding_model, interp_method)

# --------------------------
# PCA
# --------------------------
scaler = StandardScaler(with_std=False)
X = scaler.fit_transform(pca_df_filled.values.astype(float))

max_components = int(min(X.shape[0], X.shape[1]))
n_components = int(min(int(n_components_sel), max_components))
if n_components < int(n_components_sel):
    st.info(f"Reduced PCA components from {int(n_components_sel)} to {n_components} (limited by samples/features).")

pca = PCA(n_components=n_components)
PCs = pca.fit_transform(X)
pc_cols = [f"PC{i+1}" for i in range(n_components)]
PCs_df = pd.DataFrame(PCs, index=pca_df_filled.index, columns=pc_cols)

# --------------------------
# Utilities for reconstruction/forecast
# --------------------------
def reconstruct_curve_at_index(i_row: int) -> np.ndarray:
    x_centered = pca.inverse_transform(PCs[i_row].reshape(1, -1)).ravel()
    return scaler.inverse_transform(x_centered.reshape(1, -1)).ravel()

# --- FORECASTING FUNCTIONS ---
def forecast_pcs_avg_delta(PCs_matrix, window=5, pc_damp=0.5):
    if PCs_matrix.shape[0] == 0: return np.zeros((1, PCs_matrix.shape[1]))
    n, k = PCs_matrix.shape
    w = int(min(max(1, window), n))
    last = PCs_matrix[-1].reshape(1, -1)
    if w <= 1 or n == 1: avg_delta = np.zeros((k,))
    else: avg_delta = np.mean(np.diff(PCs_matrix[-w:], axis=0), axis=0)
    damp = np.ones(k)
    if k > 1: damp[1:] = float(pc_damp)
    return last + (avg_delta * damp).reshape(1, -1)

def forecast_pcs_var(PCs_df, lags=1):
    if len(PCs_df) < lags + 5:
        st.warning(f"Not enough data ({len(PCs_df)} points) for VAR({lags}). Falling back to last values.")
        return PCs_df.iloc[-1:].values
    results = VAR(PCs_df).fit(lags)
    return results.forecast(PCs_df.values[-lags:], steps=1)

def forecast_pcs_arima(PCs_df):
    forecasts = []
    for name, series in PCs_df.items():
        if len(series) < 10:
            st.warning(f"Not enough data for {name} to fit ARIMA. Using last value.")
            forecasts.append(series.iloc[-1])
            continue
        forecasts.append(ARIMA(series, order=(1, 1, 0)).fit().forecast(steps=1).iloc[0])
    return np.array(forecasts).reshape(1, -1)

# --------------------------
# Main Panel
# --------------------------
st.title("Brazilian DI Futures Yield Curve Forecast")

# --- NEW: PCA Components Heatmap ---
st.header("Principal Component Analysis Overview")
st.write("This heatmap shows how each Principal Component (PC) influences different maturities on the yield curve. This is based on the entire training data period.")
st.write("- **PC1 (Level):** Typically shows a parallel shift in the curve.")
st.write("- **PC2 (Slope):** Typically shows a steepening or flattening of the curve.")
st.write("- **PC3 (Curvature):** Typically shows a change in the curve's bow or flex.")

fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, max(4, n_components * 1.5)))
sns.heatmap(
    pca.components_,
    xticklabels=std_cols,
    yticklabels=pc_cols,
    annot=True,
    fmt=".2f",
    cmap='viridis',
    ax=ax_heatmap
)
ax_heatmap.set_title("PCA Component Loadings vs. Maturity")
st.pyplot(fig_heatmap)


# --- Get the last actual curve ---
last_actual_ts = yields_df_train.index.max()
last_actual_curve_on_std = pca_df_filled.loc[last_actual_ts].values

# --- Get the PREVIOUS day's curve for comparison ---
prev_actual_ts = None
prev_actual_curve_on_std = np.full_like(std_arr, np.nan)
if len(yields_df_train.index) > 1:
    prev_actual_ts = yields_df_train.index[-2]
    prev_actual_curve_on_std = pca_df_filled.loc[prev_actual_ts].values

# --------------------------
# In-sample reconstruction (exact date)
# --------------------------
st.markdown("---")
st.header("In-Sample Curve Reconstruction")
recon_min, recon_max = pca_df_filled.index.min().date(), pca_df_filled.index.max().date()
recon_date = st.date_input("Select date for reconstruction", value=recon_max, min_value=recon_min, max_value=recon_max, key="recon_date_pick")
recon_ts = pd.Timestamp(recon_date)

mask_pca = pca_df_filled.index.normalize() == recon_ts.normalize()
if not mask_pca.any():
    st.error(f"Selected date {recon_ts.date()} is not present in the training data. Please check your PCA Training Date range in the sidebar.")
else:
    i = int(np.where(mask_pca)[0][0])
    recon_curve_std = reconstruct_curve_at_index(i)
    orig_on_std = pca_df_filled.loc[recon_ts].values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(std_arr, recon_curve_std, marker='o', linestyle='-', color='royalblue', label=f"Reconstructed ({n_components} PCs)")
    ax.plot(std_arr, orig_on_std, marker='x', linestyle='--', color='darkorange', label="Original (interpolated on grid)")
    if show_previous_curve and prev_actual_ts is not None:
        ax.plot(std_arr, prev_actual_curve_on_std, marker='+', linestyle=':', color='green', label=f"Previous Day Curve ({prev_actual_ts.date()})")
    ax.set_xticks(std_arr)
    ax.set_xticklabels([f"{m:.2f}Y" for m in std_arr], rotation=45, ha="right")
    ax.set_xlabel("Standardized Maturity (Years)", fontsize=12)
    ax.set_ylabel("Zero Rate (%)", fontsize=12)
    ax.set_title(f"Curve Reconstruction for {recon_ts.date()} on Standard Grid", fontsize=16, pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)

# --------------------------
# Next-BD forecast
# --------------------------
st.markdown("---")
st.header(f"Next Business Day Forecast (by Contract)")

train_end_ts = pca_df_filled.index.max()
forecast_ts = next_business_day(train_end_ts, holidays_np)

# --- Select and run the chosen forecast model ---
if forecast_model_type == "None (PCA Fair Value)":
    st.write("Using **PCA Fair Value** model (no momentum or external forecast)")
    # The "Fair Value" forecast is the smooth, reconstructed curve of the last day.
    last_day_pcs_index = pca_df_filled.index.get_loc(last_actual_ts)
    pred_curve_std = reconstruct_curve_at_index(last_day_pcs_index)
else:
    if forecast_model_type == "Average Delta (Momentum)":
        st.write(f"Using **{forecast_model_type}** model")
        pcs_next = forecast_pcs_avg_delta(PCs, window=int(rolling_window_days), pc_damp=float(pc_damp))
    elif forecast_model_type == "VAR (Vector Autoregression)":
        st.write(f"Using **{forecast_model_type}** model")
        pcs_next = forecast_pcs_var(PCs_df, lags=int(var_lags))
    elif forecast_model_type == "ARIMA (per Component)":
        st.write(f"Using **{forecast_model_type}** model")
        pcs_next = forecast_pcs_arima(PCs_df)

    # Apply the predicted CHANGE to the last known actual curve
    last_pcs = PCs_df.iloc[-1].values.reshape(1, -1)
    delta_pcs = pcs_next - last_pcs
    delta_curve = pca.inverse_transform(delta_pcs)
    pred_curve_std = last_actual_curve_on_std + delta_curve.flatten()

# --- Apply Capping to momentum-based models ---
if apply_cap and cap_bps > 0 and forecast_model_type != "None (PCA Fair Value)":
    delta = pred_curve_std - last_actual_curve_on_std
    delta = np.clip(delta, -cap_frac, cap_frac)
    capped_pred_std = last_actual_curve_on_std + delta
else:
    capped_pred_std = pred_curve_std


# --- Convert curves from standard grid back to contracts ---
all_contracts = yields_df.columns
pred_contracts = std_grid_to_contracts(forecast_ts, capped_pred_std, std_arr, expiry_df, all_contracts, holidays_np, year_basis, rate_unit, compounding_model, interp_method)
last_actual_contracts = std_grid_to_contracts(last_actual_ts, last_actual_curve_on_std, std_arr, expiry_df, all_contracts, holidays_np, year_basis, rate_unit, compounding_model, interp_method)
prev_actual_contracts = std_grid_to_contracts(prev_actual_ts, prev_actual_curve_on_std, std_arr, expiry_df, all_contracts, holidays_np, year_basis, rate_unit, compounding_model, interp_method) if prev_actual_ts else None

# --- Visualization by Contract ---
fig, ax = plt.subplots(figsize=(12, 7))
valid_contracts = pred_contracts.index

ax.plot(valid_contracts, last_actual_contracts.loc[valid_contracts], marker='s', markersize=5, linestyle='-', color='gray', label=f"Last Actual ({last_actual_ts.date()})")
ax.plot(valid_contracts, pred_contracts, marker='o', markersize=7, linestyle='-', color='crimson', label=f"Predicted ({forecast_ts.date()})")
if show_previous_curve and prev_actual_contracts is not None:
    ax.plot(valid_contracts, prev_actual_contracts.loc[valid_contracts], marker='+', linestyle=':', color='green', label=f"Previous Day ({prev_actual_ts.date()})")

ax.set_xlabel("Contract Ticker", fontsize=12)
ax.set_ylabel(f"Rate ({rate_unit})", fontsize=12)
ax.set_title(f"Yield Curve Forecast for {forecast_ts.date()}", fontsize=16, pad=20)
plt.xticks(rotation=45, ha="right")
ax.legend(fontsize=10, loc='best')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
sns.despine()
plt.tight_layout()
st.pyplot(fig)

# --- Add the forecast data table by Contract ---
st.markdown("---")
st.subheader("Forecast Data by Contract")
forecast_df = pd.DataFrame({
    f"Previous Day ({prev_actual_ts.date() if prev_actual_ts else 'N/A'})": prev_actual_contracts,
    f"Last Actual ({last_actual_ts.date()})": last_actual_contracts,
    f"Predicted ({forecast_ts.date()})": pred_contracts
}).dropna(how='all')

st.dataframe(forecast_df.style.format("{:.2f}"))

# --------------------------
# Fair Value Spread Analysis
# --------------------------
st.markdown("---")
st.header("Fair Value Spread Analysis")
st.write(f"This section compares the **Last Actual** market rates against the **PCA Fair Value** for that same day ({last_actual_ts.date()}).")
st.write("A positive spread means the actual market rate is higher than the model's fair value (potentially overpriced). A negative spread means it's lower (potentially underpriced).")

# Calculate the PCA Fair Value for the last actual day
last_day_pcs_index = pca_df_filled.index.get_loc(last_actual_ts)
fair_value_std = reconstruct_curve_at_index(last_day_pcs_index)
fair_value_contracts = std_grid_to_contracts(last_actual_ts, fair_value_std, std_arr, expiry_df, all_contracts, holidays_np, year_basis, rate_unit, compounding_model, interp_method)

# Align the series and calculate the spread
aligned_actual, aligned_fv = last_actual_contracts.align(fair_value_contracts, join='inner')
spread_bps = (aligned_actual - aligned_fv) * 100 # Assuming 'Percent' unit for basis points

# Display in a chart
fig_spread, ax_spread = plt.subplots(figsize=(12, 6))
colors = ['g' if x < 0 else 'r' for x in spread_bps]
spread_bps.plot(kind='bar', ax=ax_spread, color=colors)
ax_spread.set_ylabel("Spread (Basis Points)")
ax_spread.set_xlabel("Contract Ticker")
ax_spread.set_title("Actual Market Rate vs. PCA Fair Value")
ax_spread.axhline(0, color='k', linewidth=0.8)
st.pyplot(fig_spread)

# Display in a table
spread_df = pd.DataFrame({
    "Actual Rate": aligned_actual,
    "PCA Fair Value": aligned_fv,
    "Spread (bps)": spread_bps
})
st.dataframe(spread_df.style.format("{:.2f}"))
