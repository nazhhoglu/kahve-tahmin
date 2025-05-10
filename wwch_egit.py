import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# CSV dosyasını oku
df = pd.read_csv("worldwide_coffee_habits.csv")

# Veriye genel bakış
print("İlk 5 Satır:\n", df.head())
print("\nEksik Veriler:\n", df.isnull().sum())

# Kategorik değişkenleri sayısal değerlere çevir
le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])

le_type = LabelEncoder()
df['Type of Coffee Consumed'] = le_type.fit_transform(df['Type of Coffee Consumed'])

# Özellikler ve hedef değişken
X = df[['Average Coffee Price (USD per kg)', 'Population (millions)', 'Year', 'Country']]
y = df['Coffee Consumption (kg per capita per year)']

# Eğitim/test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri tanımla
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# Eğitim ve değerlendirme
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    r2 = r2_score(y_test, predictions)
    print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Özellik önemleri (Random Forest)
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
features = X.columns

# Görselleştirme
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title("Özellik Önemleri - Random Forest")
plt.tight_layout()
plt.show()
