import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime

# 1. Data Loading and Cleaning
def load_and_clean_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Convert last_updated to datetime
    df['last_updated'] = pd.to_datetime(df['last_updated'])

    # Handle missing values
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        # Fill missing values with median for numeric columns
        df[col] = df[col].fillna(df[col].median())

    # Handle outliers using IQR method
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)

    # Normalize numeric features
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

# 2. Exploratory Data Analysis
def perform_eda(df):
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))

    # Temperature distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='temperature_celsius', bins=30)
    plt.title('Temperature Distribution')

    # Precipitation distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='condition_text', bins=30)
    plt.title('Precipitation Distribution')

    # Temperature vs Precipitation
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='temperature_celsius', y='condition_text')
    plt.title('Temperature vs Precipitation')

    # Time series of temperature
    plt.subplot(2, 2, 4)
    df.groupby('last_updated')['temperature_celsius'].mean().plot()
    plt.title('Temperature Over Time')

    plt.tight_layout()
    plt.show()

    # Correlation analysis
    correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

# 3. Time Series Forecasting
def prepare_time_series(df, target_column):
    # Aggregate data by date and calculate mean of target variable
    ts_data = df.groupby('last_updated')[target_column].mean().reset_index()
    ts_data = ts_data.set_index('last_updated')
    return ts_data

def build_forecast_model(ts_data, forecast_periods=30):
    # Fit Holt-Winters model
    model = ExponentialSmoothing(
        ts_data,
        seasonal_periods=7,  # Weekly seasonality
        trend='add',
        seasonal='add'
    ).fit()

    # Make predictions
    forecast = model.forecast(forecast_periods)

    # Calculate error metrics
    mae = mean_absolute_error(ts_data[-forecast_periods:], forecast[:forecast_periods])
    rmse = np.sqrt(mean_squared_error(ts_data[-forecast_periods:], forecast[:forecast_periods]))
    r2 = r2_score(ts_data[-forecast_periods:], forecast[:forecast_periods])

    return forecast, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# 4. Visualization of forecasts
def plot_forecast(ts_data, forecast):
    plt.figure(figsize=(12, 6))
    ts_data.plot(label='Actual')
    forecast.plot(label='Forecast', color='red')
    plt.title('Weather Forecast')
    plt.legend()
    plt.show()

# Main execution
def main(file_path):
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean_data(file_path)

    # Perform EDA
    print("\nPerforming exploratory data analysis...")
    perform_eda(df)

    # Prepare time series data for temperature forecasting
    print("\nPreparing time series data...")
    ts_data = prepare_time_series(df, 'temperature_celsius')

    # Build and evaluate forecast model
    print("\nBuilding forecast model...")
    forecast, metrics = build_forecast_model(ts_data)

    # Plot results
    print("\nPlotting forecast...")
    plot_forecast(ts_data, forecast)

    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    file_path = "/content/GlobalWeatherRepository.csv"
    main(file_path)
