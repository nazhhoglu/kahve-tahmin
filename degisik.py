import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Veriyi yükleme
data = pd.read_csv('kahve_verisi_encoded.csv')

# Kategorik verileri sayısal verilere dönüştürme (e.g. 'Country' için one-hot encoding)
data = pd.get_dummies(data, columns=['Country'])

# Özellikler ve hedef değişkeni ayırma
X = data.drop(columns=['Coffee_Consumption_kg'])
y = data['Coffee_Consumption_kg']

# Veriyi eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer Regresyon Modeli
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest Modeli
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# XGBoost Modeli
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Model Performansını Değerlendirme
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# Lineer Regresyon Sonuçları
mae_lr, mse_lr, r2_lr = evaluate_model(y_test, y_pred_lr)

# Random Forest Sonuçları
mae_rf, mse_rf, r2_rf = evaluate_model(y_test, y_pred_rf)

# XGBoost Sonuçları
mae_xgb, mse_xgb, r2_xgb = evaluate_model(y_test, y_pred_xgb)

# Sonuçları Yazdırma
print("Lineer Regresyon - MAE: ", mae_lr, "MSE: ", mse_lr, "R2: ", r2_lr)
print("Random Forest - MAE: ", mae_rf, "MSE: ", mse_rf, "R2: ", r2_rf)
print("XGBoost - MAE: ", mae_xgb, "MSE: ", mse_xgb, "R2: ", r2_xgb)
