# ⚡ BCG X Customer Churn Predictor
### BCG Data Science Job Simulation — Forage Platform

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange)
![Render](https://img.shields.io/badge/Deployed-Render.com-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🔗 Live Demo

| Resource | Link |
|---|---|
| 🚀 **Live API** | [bcgx-churn-predictor.onrender.com](https://bcgx-churn-predictor.onrender.com) |
| 📖 **Interactive API Docs** | [Try the API (Swagger UI)](https://bcgx-churn-predictor.onrender.com/docs) |
| 🌐 **Portfolio** | [aleey-lawal.github.io](https://aleey-lawal.github.io) |

> ⚠️ **Note:** The app runs on Render's free tier and may take 30–60 seconds
> to wake up after a period of inactivity.

---

## 📋 Project Overview

**PowerCo** is a major gas and electricity utility supplier to small and
medium-sized enterprises (SMEs). The company has been experiencing significant
customer churn and engaged BCG to investigate the reasons behind it.

This project builds a **machine learning pipeline** that:
- Analyses customer behaviour, consumption patterns and pricing data
- Engineers meaningful features from raw client and price datasets
- Trains a Random Forest classifier to predict customer churn
- Deploys the model as a **production-grade REST API** using FastAPI

---

## 🎯 Business Problem

> *"Is price sensitivity the primary driver of customer churn at PowerCo?"*

BCG's hypothesis was that churn is driven by price sensitivity. The analysis
explores this hypothesis using real-world style data covering customer
demographics, energy consumption, contract details and historical pricing.

---

## 📊 Dataset

| Dataset | Rows | Columns | Description |
|---|---|---|---|
| `client_data.csv` | 14,606 | 26 | Customer demographics, consumption, contract info, churn label |
| `price_data.csv` | 193,002 | 8 | Monthly off-peak, peak and mid-peak energy prices per customer |

**Churn Rate:** ~10% — a significant class imbalance that impacts recall.

---

## 🔬 Methodology

### 1. Data Cleaning & EDA
- Explored churn distribution across customer segments
- Analysed consumption patterns, tenure and pricing behaviour
- Identified class imbalance (~10% churn) as a key modelling challenge
- Converted datetime columns and handled missing values

### 2. Feature Engineering
Key features engineered from raw data:

| Feature | Description |
|---|---|
| `offpeak_diff_dec_january_energy` | Dec vs Jan off-peak variable price difference |
| `offpeak_diff_dec_january_power` | Dec vs Jan off-peak fixed price difference |
| `off_peak_peak_var_mean_diff` | Average off-peak vs peak variable price difference |
| `off_peak_peak_var_max_monthly_diff` | Maximum monthly off-peak vs peak difference |
| `tenure` | Years as a PowerCo customer |
| `months_activ` | Months since contract activation |
| `months_to_end` | Months until contract expiry |
| `months_renewal` | Months since last contract renewal |

### 3. Modelling
- **Algorithm:** Random Forest Classifier (100 trees)
- **Train/Test Split:** 75% / 25%
- **Class handling:** Standard (class imbalance acknowledged)

---

## 📈 Model Performance

| Metric | Score | Interpretation |
|---|---|---|
| **Accuracy** | 90.39% | High — but influenced by class imbalance |
| **Precision** | 85.71% | Of predicted churners, 85.71% actually churned |
| **Recall** | Low ⚠️ | Most actual churners are missed |

### ⚠️ Honest Assessment — Low Recall

The model suffers from **low recall** due to severe class imbalance (~10% churn
rate). The model sees 9 non-churners for every 1 churner during training and
defaults to predicting "no churn" for most customers.

**This limitation is displayed openly in the deployed API.**

**Proposed improvements:**
- `class_weight='balanced'` in RandomForestClassifier
- SMOTE oversampling of the minority class
- Lower classification threshold from 0.5 to 0.3

### 🔍 Top Churn Drivers (Feature Importance)
1. **Net margin** on electricity subscription
2. **12-month electricity consumption**
3. **Gross margin** on power subscription
4. **Customer tenure** and months active
5. **Price sensitivity** features (Dec–Jan differences)

> **Key finding:** While price sensitivity contributes to churn, net margin
> and consumption patterns are stronger predictors — partially challenging
> BCG's original hypothesis.

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API homepage with model metrics |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Full model performance metrics |
| `GET` | `/features` | List of all 43 model features |
| `POST` | `/predict` | Submit customer data → get churn prediction |
| `GET` | `/docs` | Interactive Swagger UI |

### Example Request

```bash
curl -X POST "https://bcgx-churn-predictor.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "id": "24011ae4ebbe3035111d65367b7f220",
       "cons_12m": 54946.0,
       "cons_gas_12m": 0.0,
       "cons_last_month": 0.0,
       "imp_cons": 0.0,
       "has_gas": "f",
       "date_activ": "2012-11-15",
       "date_end": "2016-11-15",
       "date_modif_prod": "2012-11-15",
       "date_renewal": "2015-11-15",
       "forecast_cons_12m": 4183.0,
       "forecast_discount_energy": 0.0,
       "forecast_meter_rent_12m": 1.78,
       "forecast_price_energy_off_peak": 0.095,
       "forecast_price_energy_peak": 0.0,
       "forecast_price_pow_off_peak": 44.26,
       "margin_gross_pow_ele": 268.0,
       "margin_net_pow_ele": 268.0,
       "net_margin": 309.0,
       "pow_max": 180.0,
       "nb_prod_act": 1.0,
       "num_years_antig": 3.0,
       "origin_up": "lxidpiddsbxsbosboudaobnobaocl",
       "channel_sales": "MISSING"
     }'
```

### Example Response

```json
{
  "customer_id": "24011ae4ebbe3035111d65367b7f220",
  "churn_prediction": 0,
  "churn_label": "RETAIN",
  "churn_probability": 0.331,
  "risk_level": "Medium",
  "model_note": "⚠️ This model has a low recall score — use churn_probability as a risk indicator."
}
```

---

## 📁 Project Structure

```
bcgx-churn-predictor/
├── app/
│   ├── main.py                  ← FastAPI application (5 endpoints)
│   └── templates/
│       └── index.html           ← API homepage
├── data/
│   ├── clean_client_data.csv    ← cleaned client dataset
│   └── data_for_predictions.csv ← fully engineered feature dataset
├── model/
│   ├── churn_model.pkl          ← trained Random Forest (100 trees)
│   ├── feature_names.pkl        ← ordered feature names
│   └── model_metrics.pkl        ← saved evaluation metrics
├── Notebook/
│   ├── Data Cleaning & EDA.ipynb
│   ├── Feature Engineering.ipynb
│   └── Modeling.ipynb
├── src/
│   └── pipeline.py              ← reusable preprocessing pipeline
├── render.yaml                  ← Render.com deployment config
├── requirements.txt             ← Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.11 |
| **ML Framework** | scikit-learn 1.5.1 |
| **API Framework** | FastAPI 0.111.0 |
| **Server** | Uvicorn 0.29.0 |
| **Data Processing** | Pandas 2.2.2 · NumPy 1.26.4 |
| **Model Persistence** | Joblib 1.4.2 |
| **Templating** | Jinja2 3.1.4 |
| **Deployment** | Render.com (free tier) |

---

## 🏃 Run Locally

```bash
# clone the repository
git clone https://github.com/aleey-lawal/bcgx-churn-predictor.git
cd bcgx-churn-predictor

# install dependencies
pip install -r requirements.txt

# run the API
uvicorn app.main:app --reload

# visit in browser
# http://localhost:8000       ← homepage
# http://localhost:8000/docs  ← interactive API docs
```

---

## 🎓 About This Project

This project was completed as part of the **BCG Data Science Job Simulation**
on the [Forage platform](https://www.theforage.com). The simulation provided
a hands-on opportunity to experience real-world data science workflows within
a consulting environment, covering:

- Client problem framing and hypothesis formation
- Exploratory data analysis and feature engineering
- Machine learning model development and evaluation
- Business insight communication and recommendations
- Production API deployment

---

## 👤 Author

**Lawal Lawal Ali**
- 🌐 [Portfolio](https://aleey-lawal.github.io)
- 💻 [GitHub](https://github.com/aleey-lawal)
- 💼 [LinkedIn](https://www.linkedin.com/in/aleey-lawal)
- 📧 lawalalilawal@gmail.com

---

## 📄 License

This project is licensed under the MIT License.
