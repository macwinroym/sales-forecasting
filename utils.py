import pandas as pd
import numpy as np


def create_sample_data():
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=365)

    sales = (
        200
        + np.sin(np.arange(365) * 2 * np.pi / 30) * 20
        + np.random.normal(0, 10, 365)
    )

    df = pd.DataFrame({
        'Date': dates,
        'Units_Sold': np.round(sales, 2)
    })

    return df


def load_and_preprocess(df):

    # Convert Date column
    df['Date'] = pd.to_datetime(
        df['Date'],
        dayfirst=True,
        errors='coerce'
    )

    # Rename possible sales columns
    if 'Sales' in df.columns:
        df.rename(columns={'Sales': 'Units_Sold'}, inplace=True)

    elif 'Predicted_Sales' in df.columns:
        df.rename(columns={'Predicted_Sales': 'Units_Sold'}, inplace=True)

    elif 'Revenue' in df.columns:
        df.rename(columns={'Revenue': 'Units_Sold'}, inplace=True)

    # Remove invalid dates
    df = df.dropna(subset=['Date'])

    # Sort data
    df = df.sort_values('Date')

    # Remove missing values
    df = df.dropna()

    # Reset index
    df = df.reset_index(drop=True)

    return df