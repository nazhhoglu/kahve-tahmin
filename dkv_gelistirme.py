import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Veri setini yükleyin
data = pd.read_csv('duzenlenmis_kahve_verisi.csv')

# Yıl farkı
data['Year_diff'] = 2023 - data['Year']

# Nüfusun logaritması
data['log_population'] = np.log(data['Population_Millions'])

# Kahve fiyatı standardizasyonu
scaler = StandardScaler()
data['standardized_price'] = scaler.fit_transform(data[['Coffee_Price_USD_kg']])

# Fiyat ve nüfus etkileşimi
data['price_population_interaction'] = data['Coffee_Price_USD_kg'] * data['Population_Millions']

# Kahve türü oranları
data['americano_ratio'] = data['Coffee_Type_Americano'] / data[['Coffee_Type_Americano', 'Coffee_Type_Cappuccino', 'Coffee_Type_Espresso', 'Coffee_Type_Latte', 'Coffee_Type_Mocha']].sum(axis=1)

# Yıl bazlı trend
data['yearly_trend'] = data.groupby('Year')['Coffee_Consumption_kg'].transform(lambda x: x.pct_change())

# Yıl trendi
data['year_trend'] = (data['Year'] - 2000) / (2023 - 2000)

# NaN değerlerini 0 ile doldurmak
data['yearly_trend'].fillna(0, inplace=True)

# Özellikler ve hedef değişken
X = data[['Year_diff', 'log_population', 'standardized_price', 'price_population_interaction', 'americano_ratio', 'year_trend']]
y = data['Coffee_Consumption_kg']

# Eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer Regresyon modelini eğitme
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Tahmin yapma
y_pred_lr = lr_model.predict(X_test)

# Performans değerlendirmesi
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest modelini eğitme
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Tahmin yapma
rf_pred = rf_model.predict(X_test)

# Performans değerlendirmesi
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# Sonuçları yazdırma
print(f"Lineer Regresyon - MAE: {mae_lr}, MSE: {mse_lr}, R²: {r2_lr}")
print(f"Random Forest - MAE: {rf_mae}, MSE: {rf_mse}, R²: {rf_r2}")

# Yeni CSV dosyasına kaydetme
data.to_csv('updated_coffee_consumption.csv', index=False)
print("Yeni veri dosyası 'updated_coffee_consumption.csv' olarak kaydedildi.")
