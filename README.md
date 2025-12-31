# 📈 Tea Price Forecasting with N-HiTS (Weekly, Exogenous Inputs)

A focused time-series forecasting experiment using **N-HiTS (Neural Hierarchical Interpolation for Time-Series)** to predict weekly tea leaf prices in Kolkata.  
The work centers on proper seasonality tuning and domain-relevant exogenous variables to stabilize forecasts and reduce spurious volatility.

---

## What this project does

This implementation predicts **12 weeks ahead** of tea prices using:
- Historical weekly tea leaf prices
- Monthly precipitation averages
- Monthly averaged high–low temperatures
- Fertilizer input data
- Weekly supply data

The goal is to move beyond price-only, noisy forecasts and produce smoother, more interpretable predictions aligned with known agricultural and market drivers.

📊 **Presentation on primary insights:**  (data and graph on slide 5 & 6)
https://www.canva.com/design/DAG9DN88MIg/T2D6317Hyu36hwfYUnOVqg/edit

---

## File I worked on primarily

**`experiments/long_horizon/ten.py`**

This is the primary file I authored and iterated on. It:
- Loads and cleans tea price data
- Aligns all series to a single weekly anchor
- Engineers and integrates exogenous inputs
- Tunes N-HiTS for weekly seasonality (52/26/1 hierarchy)
- Runs cross-validation and evaluation
- Produces final forecasts and plots

Other files in the repository were exploratory, baseline experiments, or adapted from existing NeuralForecast examples.

---

## How to run

```bash
python experiments/long_horizon/ten.py \
  --horizon 12 \
  --file "Kolkata (1).xlsx" \
  --weather "Kolkata_Weather_Data (2).xlsx"

  What it does (pipeline)
	•	Load: Weekly tea prices + external data sources
	•	Align: All data resampled to a consistent weekly frequency
	•	Feature context: Monthly climate, fertilizer inputs, and weekly supply
	•	Train: N-HiTS with tuned weekly seasonal hierarchy
	•	Validate: Rolling cross-validation (12-week windows)
	•	Evaluate: MAE, MSE, R²
	•	Visualize: History vs true vs predicted curves
Why N-HiTS

N-HiTS is well-suited for this problem because it:
	•	Models multiple time scales simultaneously
	•	Handles noisy auction data better than linear models
	•	Allows explicit control over seasonality structure
	•	Does not assume a fixed functional form

For this project, it was tuned specifically for weekly agricultural markets, rather than generic high-frequency time series.

⸻

Results (Kolkata)
	•	MAE: ~6.9 rupees
	•	RMSE: ~8 rupees
	•	R²: ~0.86

Given an average price around 220 rupees, this corresponds to roughly 3% average deviation, a meaningful improvement over price-only baselines.


Tech

Python • NeuralForecast • PyTorch Lightning • Pandas • NumPy • Matplotlib

⸻

Notes
	•	No future data leakage: only historical inputs are used
	•	Exogenous variables provide context, not foresight
	•	Results shown are from cross-validated forecasts, not in-sample fits
