# ML Demand Prediction – Industrial Raw Material Forecasting

Machine learning–based cumulative time series prediction of industrial raw material receivals using Prophet and XGBoost, optimized for asymmetric quantile loss (α = 0.2).

---

## 📌 Project Overview

This project was developed as part of **TDT4173 – Modern Machine Learning in Practice**, in collaboration with **Append Consulting × Hydro ASA**.

The objective is to forecast the **cumulative weight of incoming raw material deliveries (`rm_id`)** from **January 1, 2025 to May 31, 2025**, for any specified end date within this range.

Unlike traditional forecasting tasks, this problem emphasizes **conservative predictions**, as overestimating available materials may disrupt smelting operations. Therefore, evaluation is performed using **Quantile Loss at α = 0.2**, which penalizes overestimation four times more than underestimation  [oai_citation:3‡Machine_learning_task_for_TDT4173.pdf](sediment://file_00000000e010724687abf9fe47e0cb75).

The final system produces **non-decreasing cumulative forecasts** for each material, aligned with operational constraints.

---

## 📊 Dataset

The competition provides:

- `receivals.csv` – Historical raw material delivery records  
- `purchase_orders.csv` – Planned and expected deliveries  
- `materials.csv` (optional) – Raw material metadata  
- `transportation.csv` (optional) – Logistics-related information  

The task requires forecasting cumulative delivered weight per `rm_id` in kilograms.

---

## 🎯 Evaluation Metric

The evaluation metric is **Quantile Error at 0.2 (Pinball Loss)**  [oai_citation:4‡Machine_learning_task_for_TDT4173.pdf](sediment://file_00000000e010724687abf9fe47e0cb75):

\[
\text{QuantileLoss}_{0.2}(F_i, A_i) =
\max(0.2(A_i - F_i), 0.8(F_i - A_i))
\]

This asymmetric loss reflects industrial risk preferences:

- ✅ Underestimation → operationally safer  
- ❌ Overestimation → may cause production planning failures  

All modeling decisions were designed to minimize overprediction risk.

---

## 🔎 Exploratory Data Analysis

Key findings:

- Strong **weekly seasonality** (no receivals on weekends)
- Clear **holiday effects**
- Highly **zero-inflated daily weights**
- Heavy-tailed distribution of deliveries
- Few materials account for most of total production

Calendar effects were especially dominant, motivating explicit seasonal modeling  [oai_citation:5‡Report.pdf](sediment://file_000000006150720a93853791c3977cce).

---

## 🧠 Modeling Strategy

We evaluated multiple approaches before selecting the final pipeline.

### 1️⃣ Prophet (Primary Model)

Each `rm_id` is modeled independently using **Prophet**, which decomposes time series into:

- Trend
- Weekly & yearly seasonality
- Holiday effects
- Additive regressors

Prophet naturally captures structured temporal patterns and proved especially effective at modeling calendar-driven shutdowns  [oai_citation:6‡Report.pdf](sediment://file_000000006150720a93853791c3977cce).

It was used primarily for the 30 most active materials.

---

### 2️⃣ XGBoost (Alternative Model)

A global gradient-boosted tree model trained across all materials.

Features include:

- Lagged daily weights (1, 7, 14, 28 days)
- Rolling statistics (sum, mean, max)
- Zero-streak lengths
- Backlog and purchase-order aggregates
- Calendar encodings
- `rm_id` as categorical feature

XGBoost captures nonlinear cross-material interactions but required stronger postprocessing to enforce monotonic cumulative outputs  [oai_citation:7‡Report.pdf](sediment://file_000000006150720a93853791c3977cce).

---

## ⚙ Feature Engineering

Key engineered features:

- Cyclical encodings (sin/cos for day-of-week & month)
- Weekend and holiday flags
- Lagged signals
- Rolling windows
- Backlog and order aggregation features

Prophet required fewer engineered features due to its internal seasonal decomposition.

---

## 🛠 Postprocessing (Critical Step)

Raw model outputs were adjusted to satisfy operational constraints:

1. **Clipping negatives to zero**
2. **Reconstructing cumulative series**
3. **Enforcing monotonicity**
4. **Applying conservative shrinking factor**

Because Prophet minimizes MSE rather than asymmetric quantile loss, we applied a **multiplicative shrinking calibration** to bias forecasts toward underestimation  [oai_citation:8‡Report.pdf](sediment://file_000000006150720a93853791c3977cce).

This step significantly improved stability under α = 0.2.

---

## 📈 Final Pipeline

1. Data preprocessing & aggregation  
2. Feature engineering  
3. Prophet per-material forecasting  
4. XGBoost global benchmarking  
5. Conservative postprocessing  
6. Submission formatting  

The pipeline is reproducible and robust to overestimation risk.

---

## 📂 Repository Structure

```
ml-demand-prediction/
│
├── notebooks/
│   ├── Prophet_MML_cod.ipynb
│   ├── xgboostV3_cod.ipynb
│
├── data/
│   ├── receivals.csv
│   ├── purchase_orders.csv
│   ├── materials.csv
│   ├── transportation.csv
│
├── outputs/
│   └── submission.csv
├── material_receival_visualization.ipynb
├── Report.pdf
└── README.md
```

---

## 🚀 How to Reproduce

1. Clone the repository

```bash
git clone https://github.com/your-username/ml-demand-prediction.git
cd ml-demand-prediction
```

2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install required libraries

```bash
pip install pandas numpy scikit-learn xgboost prophet matplotlib seaborn
```

4. Run the notebooks in order:

- `Prophet_MML_cod.ipynb`
- `xgboostV3_cod.ipynb`

The final predictions will be generated in:

```
outputs/submission.csv
```

---

## 📌 Key Takeaways

- Industrial delivery forecasting is strongly driven by **calendar effects**.
- **Zero-inflated demand distributions** require robust modeling strategies.
- Asymmetric loss functions significantly influence model design.
- Conservative calibration improves performance under quantile objectives.
- Prophet provides strong interpretability for time series decomposition.
- XGBoost captures nonlinear cross-material dependencies.

---

## 🧠 Technical Highlights

- Expanding-window time series validation  
- Quantile loss optimization (α = 0.2)  
- Lag and rolling feature engineering  
- Backlog-based predictive signals  
- Monotonic cumulative reconstruction  
- Conservative shrinkage calibration  
- Cross-material global modeling with XGBoost  
- Per-material additive time series modeling with Prophet  

---

## 👥 Authors

**Group 153 – Italienerne**

- Sam Chhabra
- Riccardo Mazzoleni  
- Simone Tolledi  
 
NTNU – 2025

---

## 🔮 Future Improvements

- Hybrid Prophet + XGBoost stacking  
- Direct quantile regression boosting  
- Integration of external regressors (weather, logistics disruptions)  
- Learned monotonic constraints  
- Bayesian uncertainty estimation  
- Automated shrinkage calibration via cross-validation  

---

## 📄 License

Dataset sourced from a **public Kaggle competition (Append Consulting × Hydro)**.
