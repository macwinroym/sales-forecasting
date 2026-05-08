from backend.src.preprocess import load_data, clean_data
from backend.src.eda import plot_sales
from backend.src.features import create_features
from backend.src.model_arima import train_arima
from backend.src.forecast import forecast_arima

# Load data
df = load_data("data/sales.csv")
df = clean_data(df)

# EDA
plot_sales(df)

# Features
df = create_features(df)

# Train ARIMA
model = train_arima(df)

# Forecast
predictions = forecast_arima(model, 7)

print("\nNext 7 days forecast:")
print(predictions)