import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "/content/GlobalWeatherRepository.csv"
df = pd.read_csv(file_path)

### Anomaly Detection using Isolation Forest ###
outlier_features = ['temperature_celsius', 'air_quality_PM2.5', 'wind_kph']
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = isolation_forest.fit_predict(df[outlier_features])

# Visualize anomalies
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='temperature_celsius', y='air_quality_PM2.5', hue='anomaly', palette=['blue', 'red'])
plt.title("Anomaly Detection in Temperature & Air Quality")
plt.show()

### Forecasting with Multiple Models ###
df['last_updated'] = pd.to_datetime(df['last_updated'])
df = df.sort_values('last_updated')
forecast_feature = 'temperature_celsius'
train, test = train_test_split(df, test_size=0.2, shuffle=False)

# Exponential Smoothing Model
model_exp = ExponentialSmoothing(train[forecast_feature], seasonal='add', seasonal_periods=12).fit()
pred_exp = model_exp.forecast(len(test))

# Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['temperature_celsius']), df['temperature_celsius'], test_size=0.2, random_state=42)
rf_model = RandomForestRegressor()
rf_model.fit(X_train.select_dtypes(include=[np.number]), y_train)
pred_rf = rf_model.predict(X_test.select_dtypes(include=[np.number]))

# Evaluate Models
print("Exponential Smoothing MAE:", mean_absolute_error(test[forecast_feature], pred_exp))
print("Random Forest MAE:", mean_absolute_error(y_test, pred_rf))

### Climate Analysis ###
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='last_updated', y='temperature_celsius', hue='country')
plt.title("Temperature Trends by Country")
plt.show()

### Environmental Impact Analysis ###
correlation_matrix = df[['temperature_celsius', 'humidity', 'air_quality_PM2.5', 'air_quality_Ozone']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation between Air Quality and Weather Parameters")
plt.show()

### Feature Importance ###
feature_importance = pd.Series(rf_model.feature_importances_, index=X_train.select_dtypes(include=[np.number]).columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importance in Temperature Prediction")
plt.show()

### Spatial Analysis ###
geodf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
geodf.plot(column='temperature_celsius', cmap='coolwarm', legend=True)
plt.title("Geographical Temperature Distribution")
plt.show()
