# Portfolio-Optimization-using-TimeGAN-and-Transfer-Learning

## 🚀 Overview  
This project applies **deep learning** to optimize portfolios using the **top 20 FTSE 100 stocks** by market cap.  
We compare two models:  
- 🌀 **LSTM (Sen et al. architecture)**  
- ⚡ **Custom Transformer (encoder-only)**  

Predictions of weekly returns are used for **Sharpe ratio–based optimization**, generating 10 portfolios.  
The **Transformer outperforms LSTM** in both accuracy and portfolio returns.  

---

## 📂 Pipeline  
1. **Data Loading & Cleaning**  
   - Source: `yfinance` (2010–2023 prices)  
   - Outlier removal, missing value imputation  
   - Feature engineering: RSI, EMA, MoM, SMA  

2. **Feature Selection**  
   - Recursive Feature Elimination (RFE + Random Forest)  
   - Final features: Close price, RSI, SMA (5/10/20-day), Ticker  

3. **Modeling**  
   - **LSTM**: Sen et al. 2021 architecture  
   - **Transformer**: Encoder-only, Time2Vec embeddings, multi-head attention  

4. **Portfolio Optimization**  
   - Objective: **maximize Sharpe Ratio**  
   - Method: `scipy.optimize` (SLSQP)  
   - Constraints: weights ≥ 0, sum = 1  

---

## 🧠 Model Architectures
- **LSTM**: 2 layers (256 units), dropout, dense output  
- **Transformer**: Time2Vec embeddings → multi-head attention → dropout → dense output  

🔧 Optimizer: Adam (learning-rate warmup)  
📉 Loss: MAPE  

---

## 📊 Results

### 🔹 Transformer Performance  
- Avg MAPE: **0.04–0.07** (vs LSTM ~0.50)  
- Best portfolios: **5–6 assets**  
- Example: £10,000 → **£12,739 expected return**  

### 🔹 Best Portfolio (Transformer)  
| Company           | Weight (%) | Allocation (£) | Expected Return (£) |
|-------------------|------------|----------------|---------------------|
| Ferguson (FERG.L) | 28.05      | 2805           | 3574                |
| Shell (SHEL.L)    | 25.61      | 2561           | 3263                |
| Linde (LIN)       | 21.96      | 2196           | 2798                |
| BP (BP.L)         | 18.91      | 1891           | 2409                |
| Compass (CPG.L)   | 4.38       | 438            | 558                 |
| NatGrid (NG.L)    | 1.08       | 109            | 138                 |
| **Total**         | 100        | 10,000         | **12,740**          |

---

## 🏁 Conclusion & Future Work  
✔ Transformers capture **temporal dependencies** better than LSTMs  
✔ Sharpe ratio optimization → **practical portfolios**  
❌ Sentiment analysis (FinBERT + news) not integrated fully  

🔮 Next Steps:  
- Add **dynamic rebalancing**  
- Integrate **sentiment analysis**  
- Extend to multi-asset universes  

---

## ⚡ Tech Stack  
- **Data**: `yfinance`, `pandas`  
- **Models**: `TensorFlow` (LSTM, Transformer)  
- **Optimization**: `scipy.optimize` (SLSQP)  
- **Feature Selection**: `sklearn`  

---

✨ *This project shows how modern transformers outperform LSTMs in financial forecasting & portfolio construction.*  


