from neuralforecast.utils import AirPassengersDF
from utilsforecast.plotting import plot_series
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS, RNN
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("TkAgg")
import logging


Y_df = AirPassengersDF
Y_df.head()


logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

horizon = 12

# Try different hyperparmeters to improve accuracy.
models = [LSTM(input_size=2 * horizon,
               h=horizon,                    # Forecast horizon
               max_steps=500,                # Number of steps to train
               scaler_type='standard',       # Type of scaler to normalize data
               encoder_hidden_size=64,       # Defines the size of the hidden state of the LSTM
               decoder_hidden_size=64,),     # Defines the number of hidden units of each layer of the MLP decoder
          NHITS(h=horizon,                   # Forecast horizon
                input_size=2 * horizon,      # Length of input sequence
                max_steps=100,               # Number of steps to train
                n_freq_downsample=[2, 1, 1]) # Downsampling factors for each stack output
          ]
nf = NeuralForecast(models=models, freq='ME')
nf.fit(df=Y_df)

Y_hat_df = nf.predict()

# Y_hat_df = Y_hat_df
# Y_hat_df.head()

# plot_series(Y_df, Y_hat_df)
print(Y_hat_df.tail())

ax = plot_series(Y_df, Y_hat_df)

plt.tight_layout()
plt.savefig("forecast.png", dpi=200)
plt.legend()
plt.grid()
plt.plot()
plt.show()  # remove if you only want the file
print("Saved plot to forecast.png")