# 🛒 Sales Revenue Forecasting — ML Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-11557C?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**Predict retail net revenue across regions, categories & sales channels using ML regression models**

[📓 Notebook](notebooks/Sales_Forecast_ML.ipynb) · [📊 Dataset](data/sales_data.csv) · [🤖 Model](models/) · [📈 Charts](outputs/)

</div>

---

## 📌 Project Overview

This end-to-end machine learning project forecasts **net retail revenue** from structured transaction data spanning 3 years (2021–2024). It covers the complete data science pipeline — from EDA and feature engineering to model training, evaluation, and inference.

| Property | Details |
|----------|---------|
| **Domain** | Retail / Sales Analytics |
| **Task** | Regression (Revenue Prediction) |
| **Dataset** | 3,000 transactions · 17 features |
| **Best Model** | Gradient Boosting Regressor |
| **R² Score** | **0.9963** |
| **MAE** | ₹1,620 |
| **RMSE** | ₹5,086 |

---

## 📁 Project Structure

```
sales-forecast-ml/
│
├── 📂 data/
│   └── sales_data.csv          # 3,000 retail transactions (2021–2024)
│
├── 📂 notebooks/
│   └── Sales_Forecast_ML.ipynb # Full EDA + ML pipeline notebook
│
├── 📂 src/
│   └── generate_data.py        # Synthetic dataset generator
│
├── 📂 models/
│   └── best_model.pkl          # Saved Gradient Boosting model
│
├── 📂 outputs/
│   ├── chart1_monthly_trend.png
│   ├── chart2_category_channel.png
│   ├── chart3_regional.png
│   ├── chart4_heatmap.png
│   ├── chart5_model_comparison.png
│   ├── chart6_actual_vs_predicted.png
│   ├── chart7_feature_importance.png
│   └── model_results.json
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `date` | datetime | Transaction date |
| `year`, `month`, `quarter` | int | Temporal features |
| `day_of_week` | int | 0=Mon … 6=Sun |
| `region` | categorical | North / South / East / West |
| `category` | categorical | Electronics, Clothing, Groceries, etc. |
| `product` | categorical | Product name within category |
| `channel` | categorical | Online / Retail Store / Wholesale |
| `unit_price` | float | Price per unit (₹) |
| `units_sold` | int | Quantity sold |
| `discount_pct` | float | Discount applied (0–35%) |
| `revenue` | float | Gross revenue (₹) |
| `profit_margin` | float | Profit margin ratio |
| `profit` | float | Gross profit (₹) |
| `returns_pct` | float | Returns percentage |
| **`net_revenue`** | **float** | **Target: Revenue after returns (₹)** |

---

## 📈 Key Charts & Insights

### 1. Monthly Revenue Trend
![Monthly Trend](outputs/chart1_monthly_trend.png)
> **Insight**: Strong festive seasonality — revenue peaks in Oct–Dec (+40%) and dips in Feb–Mar.

---

### 2. Revenue by Category & Channel
![Category Channel](outputs/chart2_category_channel.png)
> **Insight**: Electronics dominates revenue. Wholesale channel drives highest volume in Groceries.

---

### 3. Regional Performance
![Regional](outputs/chart3_regional.png)
> **Insight**: Revenue is well-balanced across all 4 regions. North leads marginally in profit.

---

### 4. Seasonal Heatmap
![Heatmap](outputs/chart4_heatmap.png)
> **Insight**: Electronics peaks sharply in Oct–Dec. Groceries maintain steady demand year-round.

---

### 5. Model Comparison
![Model Comparison](outputs/chart5_model_comparison.png)

| Model | R² Score | MAE (₹) | RMSE (₹) |
|-------|----------|---------|----------|
| Linear Regression | 0.8663 | 18,761 | 30,748 |
| Random Forest | 0.9938 | 2,421 | 6,628 |
| **Gradient Boosting** | **0.9963** | **1,620** | **5,086** |

> **Winner**: Gradient Boosting — highest R², lowest error across both MAE and RMSE.

---

### 6. Actual vs Predicted
![Actual vs Predicted](outputs/chart6_actual_vs_predicted.png)
> Points tightly clustered around the perfect-fit diagonal, with normally distributed residuals — no systematic bias.

---

### 7. Feature Importance
![Feature Importance](outputs/chart7_feature_importance.png)
> **Top features**: `unit_price`, `units_sold`, `revenue` are the strongest predictors. Temporal features (month, quarter) capture seasonality.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ajcstanu/sales-forecast-ml.git
cd sales-forecast-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook notebooks/Sales_Forecast_ML.ipynb
```

### 4. Make a prediction
```python
import joblib, pandas as pd

model = joblib.load('models/best_model.pkl')

sample = pd.DataFrame([{
    'year': 2024, 'month': 11, 'quarter': 4, 'day_of_week': 4,
    'unit_price': 28000, 'units_sold': 12,
    'discount_pct': 0.10, 'profit_margin': 0.22, 'returns_pct': 0.03,
    'region_enc': 2,    # East
    'category_enc': 0,  # Electronics
    'channel_enc': 0,   # Online
}])

predicted = model.predict(sample)[0]
print(f'Predicted Net Revenue: ₹{predicted:,.2f}')
```

---

## 🧠 ML Pipeline

```
Raw Data (CSV)
    │
    ▼
EDA & Visualisation
    │  - Distribution analysis
    │  - Correlation matrix
    │  - Seasonal patterns
    ▼
Feature Engineering
    │  - Label encoding (region, category, channel, product)
    │  - Temporal features (year, month, quarter, day_of_week)
    ▼
Train / Test Split (80/20)
    │
    ▼
Model Training
    │  - Linear Regression (baseline)
    │  - Random Forest Regressor
    │  - Gradient Boosting Regressor  ← Best
    ▼
Evaluation (MAE · RMSE · R²)
    │
    ▼
Model Serialisation (joblib)
    │
    ▼
Inference — predict new transactions
```

---

## 🔮 Future Improvements

- [ ] Time-series forecasting with **Facebook Prophet** / ARIMA
- [ ] Hyperparameter tuning with **Optuna** / GridSearchCV
- [ ] Deploy as REST API using **FastAPI**
- [ ] Interactive dashboard with **Streamlit**
- [ ] Connect to live data pipeline (MySQL → Python → Model)
- [ ] Add cross-validation for more robust evaluation

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.11** | Core language |
| **Pandas / NumPy** | Data manipulation |
| **Scikit-learn** | ML models & evaluation |
| **Matplotlib / Seaborn** | Visualisation |
| **Jupyter Notebook** | Interactive analysis |
| **Joblib** | Model serialisation |

---

## 👩‍💻 Author

**Tanisha Gupta** — Python Developer · AI & Backend · Data Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-tanishagupta--g-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/tanishagupta-g)
[![GitHub](https://img.shields.io/badge/GitHub-ajcstanu-181717?style=flat-square&logo=github)](https://github.com/ajcstanu)
[![Email](https://img.shields.io/badge/Email-tanishaguptacse@gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:tanishaguptacse@gmail.com)

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

</div>
