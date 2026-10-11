# app/main.py
# BCG X Customer Churn Predictor — FastAPI Application
# Exposes a REST API that accepts customer data and returns churn prediction + probability

import os                           # file system operations
import joblib                       # load saved model files
import pandas as pd                 # data manipulation
import numpy as np                  # numerical operations
from datetime import datetime       # date handling

# FastAPI imports
from fastapi import FastAPI, Request, Form     # FastAPI core + form handling
from fastapi.responses import HTMLResponse     # return HTML pages
from fastapi.templating import Jinja2Templates # render HTML templates with dynamic data
from pydantic import BaseModel                 # data validation for API request bodies

# add the src/ folder to the Python path so we can import pipeline.py
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline import preprocess_input      # our feature engineering pipeline

# ── App Configuration ─────────────────────────────────────────────────────────
app = FastAPI(
    title="BCG X Customer Churn Predictor",     # shown in the API docs
    description=(
        "Predicts whether a PowerCo energy customer will churn "
        "based on their contract, consumption, and pricing data. "
        "Built as part of the BCG X Data Science Job Simulation."
    ),
    version="1.0.0"                             # API version number
)

# set up Jinja2 template engine pointing to the templates folder
templates = Jinja2Templates(directory="app/templates")

# ── Load Model and Data on Startup ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# __file__ is app/main.py → dirname = app/ → dirname again = project root

# load the trained Random Forest model from disk
model = joblib.load(os.path.join(BASE_DIR, 'model', 'churn_model.pkl'))
# this is the same model saved at the end of the modeling notebook

# load the ordered list of feature names the model was trained on
feature_names = joblib.load(os.path.join(BASE_DIR, 'model', 'feature_names.pkl'))
# critical: new data must have these exact features in this exact order

# load pre-calculated model performance metrics
model_metrics = joblib.load(os.path.join(BASE_DIR, 'model', 'model_metrics.pkl'))
# these are displayed honestly in the API responses and frontend

# load the price dataset — needed by the preprocessing pipeline
# load price data if available — use fallback if not found on server
price_data_path = os.path.join(BASE_DIR, 'data', 'clean_price_data.csv')

if not os.path.exists(price_data_path):
    # try original price data as fallback
    price_data_path = os.path.join(BASE_DIR, 'data', 'price_data.csv')

if os.path.exists(price_data_path):
    price_df = pd.read_csv(price_data_path)
    price_df['price_date'] = pd.to_datetime(
        price_df['price_date'], format='%Y-%m-%d')
    print(f"✅ Price data loaded: {price_df.shape}")
else:
    # no price data available on server — use empty DataFrame
    # pipeline.py handles this gracefully by filling price features with 0
    price_df = pd.DataFrame(columns=[
        'id', 'price_date', 'price_off_peak_var',
        'price_peak_var', 'price_mid_peak_var',
        'price_off_peak_fix', 'price_peak_fix', 'price_mid_peak_fix'
    ])
    print("⚠️ Price data not found — price features will default to 0")


# ── Request Body Schema ───────────────────────────────────────────────────────
class CustomerData(BaseModel):
    """
    Defines the expected JSON structure for a /predict API request.
    Pydantic automatically validates that all fields are present and correctly typed.
    """
    id:                          str     # unique customer identifier
    cons_12m:                    float   # electricity consumption over 12 months (kWh)
    cons_gas_12m:                float   # gas consumption over 12 months (kWh)
    cons_last_month:             float   # electricity consumption last month (kWh)
    imp_cons:                    float   # current electricity consumption (kWh)
    has_gas:                     str     # whether customer has gas subscription ('t' or 'f')
    date_activ:                  str     # contract activation date (YYYY-MM-DD)
    date_end:                    str     # contract end date (YYYY-MM-DD)
    date_modif_prod:             str     # date of last product modification (YYYY-MM-DD)
    date_renewal:                str     # date of last contract renewal (YYYY-MM-DD)
    forecast_cons_12m:           float   # forecasted electricity consumption next 12 months
    forecast_discount_energy:    float   # forecasted discount on energy
    forecast_meter_rent_12m:     float   # forecasted meter rent next 12 months
    forecast_price_energy_off_peak: float  # forecasted off-peak energy price
    forecast_price_energy_peak:  float   # forecasted peak energy price
    forecast_price_pow_off_peak: float   # forecasted off-peak power price
    margin_gross_pow_ele:        float   # gross margin on power subscription
    margin_net_pow_ele:          float   # net margin on power subscription
    net_margin:                  float   # total net margin
    pow_max:                     float   # subscribed maximum power (kW)
    nb_prod_act:                 float   # number of active products
    num_years_antig:             float   # customer seniority in years (used to compute other features)
    origin_up:                   str     # energy campaign origin code
    channel_sales:               str     # sales channel code


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Serve the HTML frontend page at the root URL.
    Returns the index.html template with model metrics injected.
    """
    # ✅ NEW way — works in Starlette 0.46.0+
    return templates.TemplateResponse(
    request=request,           # request is now a keyword argument
    name="index.html",         # template filename
    context={
        "metrics": model_metrics,           # model performance metrics
        "feature_count": len(feature_names) # number of model features
    }
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint — used by Render.com to confirm the app is running.
    Returns a simple JSON response with status information.
    """
    return {
        "status":    "healthy",                             # confirms API is running
        "model":     "RandomForestClassifier",              # model type
        "timestamp": datetime.now().isoformat(),            # current server time
        "features":  len(feature_names)                     # number of model input features
    }


@app.get("/metrics")
async def get_metrics():
    """
    Returns the model's performance metrics as JSON.
    Displayed honestly — including the low recall score.
    Useful for portfolio documentation and technical reviewers.
    """
    return {
        "model_type":    "Random Forest Classifier",
        "n_estimators":  1000,                              # number of trees
        "accuracy":      model_metrics['accuracy'],         # overall accuracy
        "precision":     model_metrics['precision'],        # precision score
        "recall":        model_metrics['recall'],           # recall score (intentionally shown even though low)
        "confusion_matrix": {
            "true_positives":  model_metrics['tp'],         # correctly predicted churners
            "true_negatives":  model_metrics['tn'],         # correctly predicted retentions
            "false_positives": model_metrics['fp'],         # wrongly predicted as churn
            "false_negatives": model_metrics['fn'],         # missed actual churners
        },
        "note": (
            f"The model correctly identifies {model_metrics['churners_caught']} out of "
            f"{model_metrics['total_churners_in_test']} actual churners in the test set. "
            f"Recall is low due to class imbalance (~10% churn rate). "
            f"Further feature engineering and class balancing (SMOTE) would improve recall."
        )
    }


@app.post("/predict")
async def predict(customer: CustomerData):
    """
    Main prediction endpoint.
    Accepts a JSON body with customer data and returns:
    - churn prediction (0 = retain, 1 = churn)
    - churn probability (0.0 to 1.0)
    - risk level label (Low / Medium / High)
    - honest note about model limitations
    """

    # convert Pydantic model to a plain Python dictionary
    data = customer.dict()

    # apply the full feature engineering pipeline to the raw input
    processed = preprocess_input(data, price_df)
    # preprocess_input returns a single-row DataFrame with all engineered features

    # reindex to ensure columns are in exactly the same order as during training
    # if any expected column is missing, fill it with 0
    processed = processed.reindex(columns=feature_names, fill_value=0)
    # reindex: aligns the DataFrame columns to match the training feature order exactly
    # fill_value=0: if a feature column is missing (e.g. rare one-hot category), fill with 0

    # generate binary prediction: 0 = not churn, 1 = churn
    prediction = int(model.predict(processed)[0])
    # model.predict returns an array — [0] extracts the single value — int() converts to integer

    # generate churn probability: value between 0.0 and 1.0
    probability = float(model.predict_proba(processed)[0][1])
    # predict_proba returns [[prob_0, prob_1]] — [0][1] extracts the probability of class 1 (churn)

    # assign a human-readable risk label based on the probability
    if probability < 0.3:
        risk_level = "Low"           # less than 30% chance of churning
    elif probability < 0.6:
        risk_level = "Medium"        # 30-60% chance of churning
    else:
        risk_level = "High"          # above 60% chance of churning

    return {
        "customer_id":       data['id'],                     # echo back the customer ID
        "churn_prediction":  prediction,                     # 0 = retain, 1 = churn
        "churn_label":       "CHURN" if prediction == 1 else "RETAIN",  # human-readable label
        "churn_probability": round(probability, 4),          # rounded to 4 decimal places
        "risk_level":        risk_level,                     # Low / Medium / High
        "model_note": (
            "⚠️ This model has a low recall score — it misses many actual churners. "
            "Use churn_probability as a risk indicator rather than a definitive prediction."
        )
    }


@app.get("/features")
async def get_features():
    """
    Returns the list of all features the model uses for prediction.
    Useful for understanding what data the API expects.
    """
    return {
        "total_features": len(feature_names),    # number of engineered features
        "features":       feature_names           # the complete ordered list
    }