# Microsoft Stock Price Prediction Using Machine Learning

I built this project to explore whether machine learning models can reliably predict Microsoft's stock price movements using historical data. Rather than just trying one approach, I tested several different models — from classical ML to deep learning — and compared how they actually performed.

---

## What This Project Does

It takes Microsoft's historical stock data (1986–2025) and tries to answer two questions:
1. Will the stock go up or down tomorrow? (classification)
2. What will the exact closing price be? (regression/forecasting)

---

## Dataset

- **Source:** Yahoo Finance via the `yfinance` API
- **Period:** 1986 to 2025
- **Features:** Open, High, Low, Close, Volume
- **Size:** 9,850+ daily records

I also engineered two key features — a `Tomorrow` column (next day's closing price) and a binary `Target` column (1 if price goes up, 0 if it goes down) — to frame both the classification and regression problems.

---

## Models I Tried

### 1. Random Forest Classifier
Used to predict the direction of the stock (up or down). I trained on everything except the last 250 rows and tested on those. Initial precision was low, so I added sliding window backtesting with a step size of 150 along with additional trend indicators like moving averages.

**After backtesting:**
| Metric | Value |
|--------|-------|
| Precision | 0.54 |
| Accuracy | 0.50 |
| F1 Score | 0.15 |

---

### 2. SVM Classifier (RBF Kernel)
Also used for directional prediction. Trained on the same split as Random Forest.

| Metric | Value |
|--------|-------|
| Accuracy | 0.47 |

---

### 3. LSTM (Deep Learning)
For continuous price forecasting. Trained with Adam optimizer and MSE loss over 20 epochs. The loss curve looked good during training, but the model struggled to capture short-term volatility in real prices.

| Metric | Value |
|--------|-------|
| MAE | 50.42 |
| RMSE | 53.56 |
| R² Score | -6.30 |
| Accuracy | 5.26% |

Honestly, LSTM underperformed here — probably needs more epochs, better features, and hyperparameter tuning to handle stock market volatility well.

---

### 4. Lasso Regression
This one surprised me. A simple linear model with L1 regularization actually outperformed LSTM significantly. The predicted prices closely followed the actual trend with minimal deviation.

| Metric | Value |
|--------|-------|
| MSE | 43.58 |
| MAE | 4.75 |
| R² Score | 0.89 |
| Accuracy | 98.42% |

---

### 5. PCA + Lasso Regression
Added PCA before Lasso to reduce dimensionality and filter noise. The predictions were smooth and closely followed the actual price movements.

| Metric | Value |
|--------|-------|
| Accuracy | 97.80% |
| Precision | 97.89% |

---

### 6. PCA Only (Reconstruction)
Used PCA purely as a reconstruction technique — no supervised model at all. Surprisingly captured the major trends but struggled with sharp market shifts.

| Metric | Value |
|--------|-------|
| MSE | 57.98 |
| MAE | 5.44 |
| R² Score | 0.85 |
| Accuracy | 97.89% |

---

## Model Comparison

| Model | Accuracy | Precision | R² Score |
|-------|----------|-----------|----------|
| Lasso Regression | 98.42% | 98.42% | 0.89 |
| PCA + Lasso | 97.80% | 97.89% | — |
| PCA Only | 97.89% | 97.89% | 0.85 |
| LSTM | 5.26% | 5.26% | -6.30 |
| Random Forest | 50% | 0.54 | — |
| SVM | 47% | — | — |

The big takeaway: simpler models with good feature engineering (Lasso, PCA+Lasso) beat deep learning here. LSTM needs a lot more tuning to be useful for stock price prediction.

---

## Tech Stack

- Python
- yfinance
- scikit-learn (Random Forest, SVM, Lasso, PCA)
- TensorFlow / Keras (LSTM)
- Matplotlib
- NumPy / Pandas

---

## Running the Notebook

```bash
git clone https://github.com/ravyavangaveti/Microsoft-Stock-Prediction-ML.git
cd Microsoft-Stock-Prediction-ML
pip install yfinance scikit-learn tensorflow matplotlib pandas numpy
jupyter notebook Microsoft_ML.ipynb
```

---

## What I Learned

Stock price prediction is genuinely hard. Deep learning doesn't automatically win — in this case, a well-tuned Lasso regression with the right features outperformed LSTM by a huge margin. Feature engineering and model simplicity matter more than model complexity for this kind of financial time-series data.

---

**Ravya Vangaveti**  
M.S. Computer Science — Data Science & Machine Learning  
University of North Carolina at Greensboro  
[GitHub](https://github.com/ravyavangaveti)
