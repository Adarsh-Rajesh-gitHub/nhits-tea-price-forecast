# run: python rnhits.py --horizon 12 --num_samples 5 --file "Kolkata (1).xlsx"
import os, argparse, logging
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.frequencies import to_offset
from ray import tune

from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE, HuberLoss
from neuralforecast.losses.numpy import mae, mse

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

def _pick_col(cols, candidates):
    """Return the first column from `cols` that matches any of `candidates` (case/space-insensitive)."""
    norm = {c.strip().lower(): c for c in cols}
    for cand in candidates:
        k = cand.strip().lower()
        if k in norm:
            return norm[k]
    return None

def load_leaf_prices(path):
    # --- Read Excel, prefer "All_Years" if present ---
    try:
        # try explicit sheet
        df = pd.read_excel(path, sheet_name="All_Years")
    except Exception:
        # fall back to first sheet
        all_sheets = pd.read_excel(path, sheet_name=None)
        first_name = next(iter(all_sheets))
        df = all_sheets[first_name]

    # --- Detect columns robustly ---
    cols = list(df.columns)
    date_col = _pick_col(cols, ["Date of Sale", "Week ending", "Week_Ending", "Date", "Week Ending"])
    price_col = _pick_col(cols, ["Leaf_Price", "Leaf Price", "LeafPrice"])

    if date_col is None:
        raise ValueError("Could not find a date column (tried: 'Date of Sale', 'Week ending').")
    if price_col is None:
        raise ValueError("Could not find a price column (tried: 'Leaf_Price').")

    # --- Parse data ---
    ds = pd.to_datetime(df[date_col], errors='coerce')
    y = pd.to_numeric(df[price_col], errors='coerce')

    s = (
        pd.DataFrame({"ds": ds, "y": y})
        .dropna(subset=["ds", "y"])
        .query("y > 0")
        .sort_values("ds")
        .drop_duplicates(subset=["ds"], keep="last")
        .reset_index(drop=True)
    )

    # --- Try to infer weekly frequency (often W-TUE for tea auctions) ---
    # We need at least 3 points for a reliable inference.
    inferred = None
    if len(s) >= 3:
        try:
            inferred = pd.infer_freq(s["ds"])
        except Exception:
            inferred = None
    freq = inferred if inferred else "W"  # sensible default if inference fails

    Y_df = s.assign(unique_id="Kolkata-Leaf")[["unique_id", "ds", "y"]]
    return Y_df, freq

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--file", type=str, default="Kolkata (1).xlsx")
    args = parser.parse_args()

    horizon = args.horizon
    num_samples = args.num_samples

    # ---- Load your pricing ----
    Y_df, freq = load_leaf_prices(args.file)
    n_time = Y_df['ds'].nunique()

    # simple splits: ~10% each, at least 12 weeks if possible
    if n_time >= 120:
        val_size  = max(12, int(0.10 * n_time))
        test_size = max(12, int(0.10 * n_time))
    else:
        val_size  = max(8, int(0.08 * n_time))
        test_size = max(8, int(0.08 * n_time))

    # ---- Model config (kept similar to yours) ----
    input_size = tune.choice([7 * horizon])  # ~7 horizons lookback (≈ 7× weeks)
    nhits_config = {
        "learning_rate": tune.loguniform(1e-5, 5e-3),
        "max_steps": tune.choice([200, 500]),
        "input_size": input_size,
        "batch_size": tune.choice([7]),
        "windows_batch_size": tune.choice([256]),
        "n_pool_kernel_size": tune.choice([[2,2,2],[16,8,1]]),
        "n_freq_downsample": tune.choice([[ (96*7)//2, 96//2, 1 ],
                                          [ (24*7)//2, 24//2, 1 ],
                                          [ 1, 1, 1 ]]),
        "dropout_prob_theta": tune.choice([0.5]),
        "activation": tune.choice(['ReLU']),
        "n_blocks": tune.choice([[1,1,1]]),
        "mlp_units": tune.choice([[[512,512],[512,512],[512,512]]]),
        "interpolation_mode": tune.choice(['linear']),
        "val_check_steps": tune.choice([100]),
        "random_seed": tune.randint(1, 10),
    }

    models = [AutoNHITS(h=horizon,
                        loss=HuberLoss(delta=0.5),
                        valid_loss=MAE(),
                        config=nhits_config,
                        num_samples=num_samples,
                        refit_with_val=True)]
    nf = NeuralForecast(models=models, freq=freq)

    # ---- Train & evaluate via temporal CV ----
    Y_hat_df = nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size, n_windows=None)

    # ---- Metrics ----
    y_true = Y_hat_df['y'].values
    y_pred = Y_hat_df['AutoNHITS'].values
    print("\n\nParsed results")
    print(f'NHITS Kolkata-Leaf h={horizon}  freq={freq}')
    print('MSE:', mse(y_pred, y_true))
    print('MAE:', mae(y_pred, y_true))

    # ---- Save forecasts ----
    out_dir = './data/KolkataLeaf'
    os.makedirs(out_dir, exist_ok=True)
    yhat_file = f'{out_dir}/{horizon}_forecasts.csv'
    Y_hat_df.to_csv(yhat_file, index=False)
    print(f"Saved forecasts to {yhat_file}")

    # ---- Plot one series ----
    first_id = Y_df.unique_id.unique()[0]
    fh = Y_hat_df[Y_hat_df.unique_id == first_id].copy()
    step = to_offset(freq)
    lookback = 4 * horizon
    hist_start = fh.ds.min() - lookback * step
    history = Y_df[(Y_df.unique_id == first_id) & (Y_df.ds >= hist_start) & (Y_df.ds < fh.ds.min())]

    plt.figure(figsize=(12,6))
    plt.plot(history.ds, history.y, label="History", color="black")
    plt.plot(fh.ds, fh['y'], label="True Future", color="green")
    plt.plot(fh.ds, fh['AutoNHITS'], label="Predicted", linestyle="--", color="blue")
    plt.title(f"NHITS Forecast (Leaf_Price weekly, h={horizon})")
    plt.xlabel("Date"); plt.ylabel("Leaf Price")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plot_path = f'{out_dir}/{horizon}_forecast_plot.png'
    plt.savefig(plot_path, dpi=200)
    plt.show()
    print(f"Saved plot to {plot_path}")