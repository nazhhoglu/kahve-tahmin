import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import numpy as np


df = pd.read_csv("veri_zenginlestirilmis.csv")


X = df.drop("Coffee_Consumption_kg", axis=1)  # 'target' yerine hedef sütun adını yaz
y = df["Coffee_Consumption_kg"]

# Train/Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Tahminler
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Performans ölçütleri
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"Train R² Skoru: {train_r2:.4f}")
print(f"Test R² Skoru: {test_r2:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
