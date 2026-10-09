# src/pipeline.py
# Reusable preprocessing pipeline for the BCG X Churn Predictor API

import pandas as pd            # data manipulation
import numpy as np             # numerical operations
from datetime import datetime  # date handling

# reference date used for all date-to-month conversions
REFERENCE_DATE = datetime(2016, 1, 1)

# columns that need log transformation (identified as skewed during EDA)
SKEWED_COLS = [
    'cons_12m',
    'cons_gas_12m',
    'cons_last_month',
    'forecast_cons_12m',
    'forecast_discount_energy',
    'forecast_meter_rent_12m',
    'forecast_price_energy_off_peak',
    'forecast_price_energy_peak',
    'forecast_price_pow_off_peak'
]


def convert_months(reference_date, date_value):
    """
    Calculate number of whole months between reference date and a given date.
    """
    year_diff  = reference_date.year  - date_value.year
    month_diff = reference_date.month - date_value.month
    months = year_diff * 12 + month_diff
    if reference_date.day < date_value.day:
        months -= 1
    return months


def preprocess_input(data: dict, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a single customer input.
    Mirrors every step from the feature engineering notebook.
    """

    # Step 1: Convert input dict to single-row DataFrame
    df = pd.DataFrame([data])

    # Step 2: Convert date strings to datetime
    for col in ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal']:
        df[col] = pd.to_datetime(df[col], format='mixed')
        # format='mixed': pandas auto-detects date format — prevents crashes

    # Step 3: Compute tenure (years as a customer)
    df['tenure'] = ((df['date_end'] - df['date_activ']).dt.days // 365)
    # integer division by 365 gives complete years

    # Step 4: Convert dates to month-based numeric features
    date_activ      = df['date_activ'].iloc[0]
    date_end        = df['date_end'].iloc[0]
    date_modif_prod = df['date_modif_prod'].iloc[0]
    date_renewal    = df['date_renewal'].iloc[0]

    df['months_activ']      = convert_months(REFERENCE_DATE, date_activ)
    df['months_to_end']     = -convert_months(REFERENCE_DATE, date_end)
    df['months_modif_prod'] = convert_months(REFERENCE_DATE, date_modif_prod)
    df['months_renewal']    = convert_months(REFERENCE_DATE, date_renewal)

    # drop raw date columns — no longer needed after conversion
    df = df.drop(columns=['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'])

    # Step 5: Encode has_gas boolean to 0/1
    df['has_gas'] = 1 if df['has_gas'].iloc[0] in ['t', True, 1, '1'] else 0

    # Step 6: Price difference features
    customer_id = data.get('id')
    customer_prices = price_df[price_df['id'] == customer_id].copy()
    customer_prices['price_date'] = pd.to_datetime(customer_prices['price_date'])

    if len(customer_prices) > 0:
        # Dec vs Jan off-peak price differences
        monthly = customer_prices.groupby('price_date').agg({
            'price_off_peak_var': 'mean',
            'price_off_peak_fix': 'mean'
        }).reset_index()

        jan = monthly.iloc[0]
        dec = monthly.iloc[-1]

        df['offpeak_diff_dec_january_energy'] = dec['price_off_peak_var'] - jan['price_off_peak_var']
        df['offpeak_diff_dec_january_power']  = dec['price_off_peak_fix'] - jan['price_off_peak_fix']

        # average price differences between time periods
        mean_p = customer_prices.agg({
            'price_off_peak_var': 'mean',
            'price_peak_var':     'mean',
            'price_mid_peak_var': 'mean',
            'price_off_peak_fix': 'mean',
            'price_peak_fix':     'mean',
            'price_mid_peak_fix': 'mean'
        })

        df['off_peak_peak_var_mean_diff']     = mean_p['price_off_peak_var'] - mean_p['price_peak_var']
        df['peak_mid_peak_var_mean_diff']     = mean_p['price_peak_var']     - mean_p['price_mid_peak_var']
        df['off_peak_mid_peak_var_mean_diff'] = mean_p['price_off_peak_var'] - mean_p['price_mid_peak_var']
        df['off_peak_peak_fix_mean_diff']     = mean_p['price_off_peak_fix'] - mean_p['price_peak_fix']
        df['peak_mid_peak_fix_mean_diff']     = mean_p['price_peak_fix']     - mean_p['price_mid_peak_fix']
        df['off_peak_mid_peak_fix_mean_diff'] = mean_p['price_off_peak_fix'] - mean_p['price_mid_peak_fix']

        # maximum monthly price differences
        monthly_all = customer_prices.groupby('price_date').agg({
            'price_off_peak_var': 'mean',
            'price_peak_var':     'mean',
            'price_mid_peak_var': 'mean',
            'price_off_peak_fix': 'mean',
            'price_peak_fix':     'mean',
            'price_mid_peak_fix': 'mean'
        }).reset_index()

        df['off_peak_peak_var_max_monthly_diff']     = (monthly_all['price_off_peak_var'] - monthly_all['price_peak_var']).max()
        df['peak_mid_peak_var_max_monthly_diff']     = (monthly_all['price_peak_var']     - monthly_all['price_mid_peak_var']).max()
        df['off_peak_mid_peak_var_max_monthly_diff'] = (monthly_all['price_off_peak_var'] - monthly_all['price_mid_peak_var']).max()
        df['off_peak_peak_fix_max_monthly_diff']     = (monthly_all['price_off_peak_fix'] - monthly_all['price_peak_fix']).max()
        df['peak_mid_peak_fix_max_monthly_diff']     = (monthly_all['price_peak_fix']     - monthly_all['price_mid_peak_fix']).max()
        df['off_peak_mid_peak_fix_max_monthly_diff'] = (monthly_all['price_off_peak_fix'] - monthly_all['price_mid_peak_fix']).max()

    else:
        # no price data found for this customer — fill with 0
        price_diff_cols = [
            'offpeak_diff_dec_january_energy',
            'offpeak_diff_dec_january_power',
            'off_peak_peak_var_mean_diff',
            'peak_mid_peak_var_mean_diff',
            'off_peak_mid_peak_var_mean_diff',
            'off_peak_peak_fix_mean_diff',
            'peak_mid_peak_fix_mean_diff',
            'off_peak_mid_peak_fix_mean_diff',
            'off_peak_peak_var_max_monthly_diff',
            'peak_mid_peak_var_max_monthly_diff',
            'off_peak_mid_peak_var_max_monthly_diff',
            'off_peak_peak_fix_max_monthly_diff',
            'peak_mid_peak_fix_max_monthly_diff',
            'off_peak_mid_peak_fix_max_monthly_diff'
        ]
        for col in price_diff_cols:
            df[col] = 0

    # Step 7: Log transform skewed columns
    for col in SKEWED_COLS:
        if col in df.columns:
            df[col] = np.log10(df[col] + 1)
            # add 1 before log to avoid log(0) = undefined

    # Step 8: Drop non-feature columns
    drop_cols = ['id', 'churn', 'num_years_antig', 'forecast_cons_year']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df