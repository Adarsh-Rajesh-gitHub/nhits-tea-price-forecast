# file with working model
import os, argparse, logging
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neuralforecast.core import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE, HuberLoss
from neuralforecast.losses.numpy import mae, mse

# --------- Utils ---------
def _pick_col(cols, candidates):
    norm = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        k = cand.strip().lower()
        if k in norm: return norm[k]
    return None

def load_leaf_prices(path):
    try:
        df = pd.read_excel(path, sheet_name="All_Years")
    except Exception:
        sheets = pd.read_excel(path, sheet_name=None)
        df = sheets[next(iter(sheets))]
    date_col  = _pick_col(df.columns, ["Date of Sale","Week ending","Week_Ending","Date","Week Ending"])
    price_col = _pick_col(df.columns, ["Leaf_Price","Leaf Price","LeafPrice"])
    if date_col is None or price_col is None:
        raise ValueError("Missing date or price column.")
    ds = pd.to_datetime(df[date_col], errors="coerce")
    y  = pd.to_numeric(df[price_col], errors="coerce")
    s = (pd.DataFrame({"ds": ds, "y": y})
           .dropna(subset=["ds","y"])
           .query("y > 0")
           .sort_values("ds")
           .drop_duplicates(subset=["ds"], keep="last")
           .reset_index(drop=True))
    # Infer anchor weekday and use it consistently
    dow3 = s["ds"].dt.day_name().str[:3].str.upper().mode()[0]
    freq_anchor = f"W-{dow3}"
    s = (s.set_index("ds").resample(freq_anchor).last().ffill().reset_index())  # ensure strict weekly grid
    Y_df = s.assign(unique_id="Kolkata-Leaf")[["unique_id","ds","y"]]
    return Y_df, freq_anchor

def load_weather(path, freq_anchor):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx",".xls")) else pd.read_csv(path)
    date_c = _pick_col(df.columns, ["Date","date"])
    if date_c is None: raise ValueError("Weather file missing Date column.")
    date = pd.to_datetime(df[date_c], errors="coerce")

    rain = pd.to_numeric(df.get(_pick_col(df.columns, ["Rain","rain","precip","precip_mm"])), errors="coerce")
    tmax = pd.to_numeric(df.get(_pick_col(df.columns, ["Temp Max","Temp_Max","tmax","tmax_c"])), errors="coerce")
    tmin = pd.to_numeric(df.get(_pick_col(df.columns, ["Temp Min","Temp_Min","tmin","tmin_c"])), errors="coerce")

    w = (pd.DataFrame({"ds": date, "rain": rain, "tmax": tmax, "tmin": tmin})
           .dropna(subset=["ds"])
           .set_index("ds")
           .resample(freq_anchor)
           .agg({"rain":"sum","tmax":"mean","tmin":"mean"})
           .reset_index())
    return w  # ds, rain, tmax, tmin

def make_calendar(ds):
    cal = pd.DataFrame({"ds": ds})
    cal["woy"]   = cal["ds"].dt.isocalendar().week.astype(int)
    cal["month"] = cal["ds"].dt.month.astype(int)
    return cal

def add_weather_lagged_hist_exogs(Y_df, W):
    # Merge and build LAGGED/ROLLING features only (strictly historical → no future leakage)
    z = Y_df.merge(W, on="ds", how="left").sort_values("ds")
    # Fill gaps *historically* by carrying last observation backward-safe
    for c in ["rain","tmax","tmin"]:
        if c in z.columns:
            z[c] = z[c].ffill()

    # Create lag/rolling features that are known at time t
    # 4-week rolling means + 1-week lag to avoid peeking current-week weather
    z["rain_4w"] = z["rain"].rolling(4, min_periods=4).mean().shift(1)
    z["tmax_4w"] = z["tmax"].rolling(4, min_periods=4).mean().shift(1)
    z["tmin_4w"] = z["tmin"].rolling(4, min_periods=4).mean().shift(1)

    hist_cols = ["rain_4w","tmax_4w","tmin_4w"]
    z = z.dropna(subset=hist_cols)  # ensure valid windows only
    return z[["unique_id","ds","y"] + hist_cols], hist_cols

def scale_hist_exogs(df, hist_cols, val_size):
    # Standardize hist exogs using TRAIN stats only (leave last val_size as validation)
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[-val_size:] = False  # last val_size rows are validation period
    train = df.loc[train_mask, hist_cols]
    means = train.mean()
    stds  = train.std().replace(0, 1.0)
    df.loc[:, hist_cols] = (df[hist_cols] - means) / stds
    return df

# --------- Main ---------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--file", type=str, required=True)
    p.add_argument("--weather", type=str, default=None)
    args = p.parse_args()

    # Load target & consistent weekly anchor
    Y_df, freq_anchor = load_leaf_prices(args.file)
    n_time = Y_df["ds"].nunique()
    val_size = max(12, int((0.10 if n_time >= 120 else 0.08) * n_time))

    # Calendar (future-known) exogs
    cal = make_calendar(Y_df["ds"])

    # Historical exogs: lagged/rolling weather (no arbitrary 25-week shift; no future leakage)
    if args.weather:
        W = load_weather(args.weather, freq_anchor=freq_anchor)
        Y_hist, hist_cols = add_weather_lagged_hist_exogs(Y_df, W)
    else:
        Y_hist, hist_cols = Y_df.copy(), []

    # Merge calendar features (future exogs)
    YX_df = (Y_hist.merge(cal, on="ds", how="left")
                    .sort_values("ds")
                    .reset_index(drop=True))

    futr_cols = ["woy","month"]

    # Scale hist exogs using train-only stats (simple z-score)
    if hist_cols:
        YX_df = scale_hist_exogs(YX_df, hist_cols, val_size=val_size)

    # ---------- Models (identical freq; weekly-appropriate params) ----------
    common_kwargs = dict(
        h=args.horizon,
        input_size=7*args.horizon,
        loss=HuberLoss(delta=0.5),
        valid_loss=MAE(),
        learning_rate=1e-3,
        max_steps=5000,
        batch_size=4,                 # shorter weekly series → smaller batch
        windows_batch_size=64,        # avoid oversized window sampling
        n_pool_kernel_size=[2,2,2],
        n_freq_downsample=[52, 26, 1],# weekly seasonality, not high-frequency nonsense
        dropout_prob_theta=0.5,
        activation="ReLU",
        n_blocks=[1,1,1],
        mlp_units=[[512,512],[512,512],[512,512]],
        interpolation_mode="linear",
        val_check_steps=100,
        random_seed=5
    )

    model_noX = NHITS(**common_kwargs)
    model_X   = NHITS(**common_kwargs, hist_exog_list=hist_cols, futr_exog_list=futr_cols)

    # ---------- Cross-Validation (identical setup) ----------
    nf_noX = NeuralForecast(models=[model_noX], freq=freq_anchor)
    nf_X   = NeuralForecast(models=[model_X],   freq=freq_anchor)

    # CV for no-exog
    Y_hat_noX = nf_noX.cross_validation(df=YX_df[["unique_id","ds","y"]], val_size=val_size, n_windows=1)
    mae_noX = mae(Y_hat_noX['NHITS'].values, Y_hat_noX['y'].values)

    # CV for exog
    Y_hat_X = nf_X.cross_validation(df=YX_df, val_size=val_size, n_windows=1)
    mae_X = mae(Y_hat_X['NHITS'].values, Y_hat_X['y'].values)
    mse_X = mse(Y_hat_X['NHITS'].values, Y_hat_X['y'].values)

    print(f"MAE (no exogs): {mae_noX:.4f}")
    print(f"MAE (with exogs): {mae_X:.4f}")
    print(f"MSE (with exogs): {mse_X:.4f}")

    # ---------- Final fit & forecast ----------
    # Fit exog model on full data
    nf_final = NeuralForecast(models=[model_X], freq=freq_anchor)
    nf_final.fit(df=YX_df)

    # Build future calendar (futr exogs only); hist exogs are past-only (OK for NHITS)
    last_ds = YX_df["ds"].max()
    future_ds = pd.date_range(last_ds, periods=args.horizon+1, freq=freq_anchor, inclusive="right")
    futr_cal = make_calendar(future_ds)
    futr_df = pd.DataFrame({
        "unique_id": YX_df["unique_id"].iloc[0],
        "ds": futr_cal["ds"],
        "woy": futr_cal["woy"],
        "month": futr_cal["month"]
    })

    Y_fcst = nf_final.predict(futr_df=futr_df)

    # ---------- Plot CV curve for the exog model ----------
    first_id = YX_df.unique_id.unique()[0]
    fh = Y_hat_X[Y_hat_X.unique_id == first_id].copy()
    g = fh.groupby("ds", as_index=False).agg(y=("y","mean"), yhat=("NHITS","mean"))
    plt.figure(figsize=(12,6))
    plt.plot(g.ds, g.y, label="True")
    plt.plot(g.ds, g.yhat, "--", label="Pred (CV)")
    plt.title(f"NHITS weekly CV (h={args.horizon})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # ---------- Print head of forecast ----------
    print("\nForecast head (with calendar futr exogs only):")
    print(Y_fcst.head())