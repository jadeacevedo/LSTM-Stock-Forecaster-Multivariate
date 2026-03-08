## Project Overview
The primary objective was to develop a robust prediction system capable of capturing the intricate patterns and temporal dependencies inherent in financial markets. While traditional models often struggle with the volatility of stock data, this project leverages deep learning architectures specifically designed for sequential data.

## 1. Data Acquisition & Feature Engineering
Image Reference: <img width="1528" height="1082" alt="image" src="https://github.com/user-attachments/assets/d2a006d1-cb86-4fb0-ac6d-e77938bd47c1" />

To build a foundation for the models, we utilized synthetic data modeled after Apple Inc. (AAPL) stock behavior using Geometric Brownian Motion.

Close Price & Bollinger Bands: We visualized the "Close" price alongside EMA-10 and EMA-50 to identify trends. Bollinger Bands were calculated to visualize market volatility.

Relative Strength Index (RSI-14): A momentum oscillator was implemented to identify overbought (above 70) and oversold (below 30) conditions, providing the model with critical market context.

Trading Volume: Volume data was included to represent market activity levels for each trading day.


## 2. Neural Network Architectures
Image Reference: <img width="1748" height="544" alt="image" src="https://github.com/user-attachments/assets/417429df-d854-491f-8823-219e3b67f81a" />


Two distinct architectures were implemented from scratch to compare their ability to handle time-series data:

Simple RNN (Unrolled): This model processes inputs sequentially, maintaining a hidden state h(t) that carries information from previous time steps. However, it is susceptible to the vanishing gradient problem, making long-term dependencies difficult to capture.

LSTM Gate Architecture: To mitigate RNN limitations, the LSTM introduces a Cell State C(t) and three specific gates: Forget, Input, and Output. These gates allow the model to selectively retain or discard information over long periods.


## 3. Training and Convergence
Image Reference: <img width="1638" height="435" alt="image" src="https://github.com/user-attachments/assets/7ed14ca9-3ba9-44bf-9d55-787a06223076" />


We monitored the Mean Squared Error (MSE) Loss over 40 epochs for three model variations:

Training Dynamics: All models showed rapid convergence within the first 10 epochs.

Validation: The gap between the training (solid) and validation (dashed) lines remained narrow, suggesting that the models generalized well without significant overfitting to the synthetic training set.


## 4. Model Predictions vs. Actuals
Image Reference: blob:https: <img width="1527" height="1298" alt="image" src="https://github.com/user-attachments/assets/696c0aab-3cfc-4d6b-9991-819cb236199c" />


We evaluated performance on a test set covering late 2022 through early 2023:

Simple RNN: Surprisingly achieved the highest R 
2
  (0.8708) and lowest RMSE ($2.46). This indicates that for this specific dataset, the simpler recurrent structure was highly efficient at following immediate price fluctuations.

LSTM (Univariate & Multivariate): Both LSTM models showed higher error rates (RMSE ~$3.91-$3.94). While LSTMs are theoretically superior for long-term dependencies, they may require more complex data or hyperparameter tuning to outperform a Simple RNN on localized price movements.


## 5. Residual Analysis
Image Reference: <img width="1637" height="759" alt="image" src="https://github.com/user-attachments/assets/4d98d174-51ae-457b-8ddc-dce6c844ae0b" />


To validate model reliability, we analyzed the residuals (Error = Actual - Predicted):

Error Distribution: The Simple RNN's error distribution is centered near zero (μ=0.39), suggesting a low bias.

Residual Patterns: The LSTM residuals showed more distinct "waves," indicating that some temporal patterns in the data were not fully captured by the current LSTM configurations, leading to slightly higher systematic error.


## 6. Final Model Comparison
Image Reference: <img width="1418" height="436" alt="image" src="https://github.com/user-attachments/assets/8c6895ff-5bda-4703-abc1-643b0753df1c" />


The final evaluation used three primary metrics: RMSE (Root Mean Square Error), MAE (Mean Absolute Error), and MAPE (Mean Absolute Percentage Error).

Model	RMSE	MAE	MAPE (%)	R 
2
 
Simple RNN	2.4566	2.0005	1.1538	0.8708
LSTM (Univariate)	3.9126	3.0285	1.7615	0.6722
LSTM (Multivariate)	3.9406	3.2260	1.8641	0.6674
### Conclusion

While LSTMs are designed to solve the "memory" issues of standard RNNs, the Simple RNN proved most effective for this specific stock price prediction task. This highlights the importance of matching model complexity to the specific characteristics of the dataset.

