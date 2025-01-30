# PMAcceleratorAssesment

## Overview
This project implements a comprehensive weather analysis system that combines data preprocessing, exploratory data analysis (EDA), anomaly detection, and multiple forecasting models to analyze global weather patterns and environmental conditions.

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Time series forecasting
- Anomaly detection
- Environmental impact analysis
- Spatial analysis
- Feature importance analysis

## Requirements
- Python 3.x
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - statsmodels
  - geopandas

## Project Structure
```
.
├── assesment.py        # Main analysis script
└── README.md           # Project documentation
```

## Methodology

### 1. Data Loading and Cleaning
- Handles missing values using median imputation
- Removes outliers using the IQR method
- Normalizes numeric features using StandardScaler
- Converts timestamps to datetime format

### 2. Exploratory Data Analysis
- Temperature distribution analysis
- Precipitation patterns
- Temperature vs. precipitation correlation
- Time series visualization
- Correlation matrix analysis

### 3. Time Series Forecasting
- Implements Holt-Winters Exponential Smoothing
- Includes trend and seasonal components
- Provides forecasting for 30 periods ahead
- Calculates performance metrics (MAE, RMSE, R²)

### 4. Advanced Analytics
#### Anomaly Detection
- Uses Isolation Forest algorithm
- Focuses on temperature, air quality (PM2.5), and wind speed
- Contamination rate: 5%

#### Multiple Forecasting Models
- Exponential Smoothing
- Random Forest Regression
- Model comparison using MAE

#### Environmental Impact Analysis
- Correlation analysis between weather parameters
- Air quality impact assessment
- Temperature trend analysis by country

#### Spatial Analysis
- Geographical temperature distribution
- Uses latitude/longitude data
- Visual mapping of weather patterns

## Usage

1. Data Preparation:
```python
df = load_and_clean_data(file_path)
```

2. Run EDA:
```python
perform_eda(df)
```

3. Generate Forecasts:
```python
ts_data = prepare_time_series(df, 'temperature_celsius')
forecast, metrics = build_forecast_model(ts_data)
```

4. Visualize Results:
```python
plot_forecast(ts_data, forecast)
```

## Performance Metrics
The system evaluates forecasting performance using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²)

## Data Requirements
Input CSV file should contain the following columns:
- last_updated
- temperature_celsius
- condition_text
- air_quality_PM2.5
- air_quality_Ozone
- humidity
- wind_kph
- latitude
- longitude
- country

## Limitations and Considerations
- The system assumes weekly seasonality in the time series forecasting
- Anomaly detection is configured with a fixed contamination rate
- Spatial analysis requires valid geographical coordinates
- Performance may vary based on data quality and completeness
