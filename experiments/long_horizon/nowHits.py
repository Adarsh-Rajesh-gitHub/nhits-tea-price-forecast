# run: python nowHits.py --horizon 12 --num_samples 5 --file "Kolkata (1).xlsx" --weather "Kolkata_Weather_Data (2).xlsx"
import os, argparse, logging
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast.core import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE, HuberLoss
from neuralforecast.losses.numpy import mae, mse

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

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
    try: freq = pd.infer_freq(s["ds"])
    except Exception: freq = None
    freq = freq or "W"   # default weekly
    Y_df = s.assign(unique_id="Kolkata-Leaf")[["unique_id","ds","y"]]
    return Y_df, freq

def load_weather(path, freq_anchor="W"):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx",".xls")) else pd.read_csv(path)
    date = pd.to_datetime(df[_pick_col(df.columns, ["Date","date"])], errors="coerce")
    rain = pd.to_numeric(df.get(_pick_col(df.columns, ["Rain","rain"])), errors="coerce")
    tmax = pd.to_numeric(df.get(_pick_col(df.columns, ["Temp Max","Temp_Max","tmax"])), errors="coerce")
    tmin = pd.to_numeric(df.get(_pick_col(df.columns, ["Temp Min","Temp_Min","tmin"])), errors="coerce")
    w = (pd.DataFrame({"ds": date, "rain": rain, "tmax": tmax, "tmin": tmin})
           .dropna(subset=["ds"])
           .set_index("ds")
           .resample(freq_anchor).agg({"rain":"sum","tmax":"mean","tmin":"mean"})
           .reset_index())
    return w  # ds,rain,tmax,tmin

def make_calendar(ds):
    cal = pd.DataFrame({"ds": ds})
    cal["woy"]   = cal["ds"].dt.isocalendar().week.astype(int)
    cal["month"] = cal["ds"].dt.month.astype(int)
    return cal

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--num_samples", type=int, default=5)   # kept for CLI parity
    p.add_argument("--file", type=str, required=True)
    p.add_argument("--weather", type=str, default=None)
    args = p.parse_args()

    # --- Load target & infer native weekly freq ---
    Y_df, freq = load_leaf_prices(args.file)
    n_time = Y_df["ds"].nunique()
    val_size = max(12, int((0.10 if n_time >= 120 else 0.08) * n_time))

    # Compute the auction week anchor (e.g., W-TUE) and use it consistently
    dow3 = Y_df["ds"].dt.day_name().str[:3].str.upper().mode()[0]
    freq_anchor = f"W-{dow3}"

    # --- Build features inside df so NHITS can find them ---
    YX_df = Y_df.copy()
    if args.weather:
        W   = load_weather(args.weather, freq_anchor=freq_anchor)
        W[['rain','tmax','tmin']] = W[['rain','tmax','tmin']].shift(-25)
        cal = make_calendar(Y_df["ds"])
        YX_df = (YX_df.merge(W, on="ds", how="left")
                       .merge(cal, on="ds", how="left")
                       .sort_values("ds"))
        for c in ("rain","tmax","tmin"):
            if c in YX_df.columns:
                YX_df[c] = YX_df[c].ffill()

    hist_feats = [c for c in ["rain","tmax","tmin"] if c in YX_df.columns]
    futr_feats = [c for c in ["woy","month"]        if c in YX_df.columns]
   
    # --- Model ---
    model = NHITS(
        h=args.horizon,
        input_size=7*args.horizon,
        loss=HuberLoss(delta=0.5), valid_loss=MAE(),
        learning_rate=1e-3, max_steps=500,
        batch_size=7, windows_batch_size=256,
        n_pool_kernel_size=[2,2,2],
        n_freq_downsample=[(96*7)//2, 96//2, 1],
        dropout_prob_theta=0.5, activation="ReLU",
        n_blocks=[1,1,1],
        mlp_units=[[512,512],[512,512],[512,512]],
        interpolation_mode="linear",
        val_check_steps=100, random_seed=5,
        hist_exog_list=hist_feats, futr_exog_list=futr_feats,
    )
    model_noX = NHITS(h=args.horizon, input_size=7*args.horizon, loss=HuberLoss(delta=0.5),
                  valid_loss=MAE(), learning_rate=1e-3, max_steps=500, batch_size=7,
                  windows_batch_size=256, n_pool_kernel_size=[2,2,2],
                  n_freq_downsample=[(96*7)//2,96//2,1], dropout_prob_theta=0.5,
                  activation='ReLU', n_blocks=[1,1,1], mlp_units=[[512,512],[512,512],[512,512]],
                  interpolation_mode='linear', val_check_steps=100, random_seed=5)
    nf2 = NeuralForecast(models=[model_noX], freq=freq)
    Y_hat2 = nf2.cross_validation(df=Y_df, val_size=val_size, n_windows=1)
    print("MAE no exogs:", mae(Y_hat2['NHITS'].values, Y_hat2['y'].values))
    
    nf = NeuralForecast(models=[model], freq=freq_anchor)

    # --- CV: pick ONE of test_size or n_windows ---
    Y_hat_df = nf.cross_validation(df=YX_df, val_size=val_size, n_windows=1)

    # --- Metrics & plot ---
    y_true = Y_hat_df["y"].values
    y_pred = Y_hat_df["NHITS"].values
    print("MSE:", mse(y_pred, y_true))
    print("MAE:", mae(y_pred, y_true))

    first_id = Y_df.unique_id.unique()[0]
    fh = Y_hat_df[Y_hat_df.unique_id == first_id].copy()
    g = fh.groupby("ds", as_index=False).agg(y=("y","mean"), yhat=("NHITS","mean"))
    plt.figure(figsize=(12,6))
    plt.plot(g.ds, g.y, label="True")
    plt.plot(g.ds, g.yhat, "--", label="Pred")
    plt.title(f"NHITS weekly (h={args.horizon})"); plt.grid(True); plt.legend(); plt.tight_layout()