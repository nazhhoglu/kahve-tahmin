import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import lightgbm as lgb

# Veriyi oku
df = pd.read_csv("veri_zenginlestirilmis.csv")

# Özellikler ve hedef değişken
X = df.drop(columns=['Coffee_Consumption_kg'])
y = df['Coffee_Consumption_kg']

# Eğitim ve test seti ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri başlat
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Support Vector Regressor': SVR(),
    'XGBoost': XGBRegressor(),
    'LightGBM': lgb.LGBMRegressor()
}

# Model sonuçları
results = {}

# Eğitim ve test hatalarını hesapla
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}

# **Adım 1: Model Karşılaştırma Grafiği**
# R² skorlarının grafiği
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=[result['R2'] for result in results.values()])
plt.title('Model Karşılaştırması (R² Skoru)')
plt.ylabel('R² Skoru')
plt.xlabel('Modeller')
plt.xticks(rotation=45)
plt.show()

# **Adım 2: Önemli Özelliklerin Gösterilmesi**
# LightGBM modelinin özellik önemini gösterelim
lightgbm_model = models['LightGBM']
lightgbm_model.fit(X_train, y_train)
importances = lightgbm_model.feature_importances_

# Özellikler ile birlikte gösterme
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Özelliklerin grafiği
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('LightGBM Özellik Önemlilikleri')
plt.show()

# **Adım 3: Gerçek ve Tahmin Edilen Değerlerin Karşılaştırılması**
y_preds = {name: model.predict(X_test) for name, model in models.items()}

plt.figure(figsize=(12, 8))
for i, (name, y_pred) in enumerate(y_preds.items(), 1):
    plt.subplot(2, 3, i)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title(f'{name} Gerçek vs Tahmin')
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Edilen Değerler')
plt.tight_layout()
plt.show()

# **Adım 4: Test ve Train Ayrımı ile Model Değerlendirmesi (Cross-Validation)**
cv_results = {}
for name, model in models.items():
    cv_score = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_results[name] = cv_score.mean()

# Cross-validation sonuçlarının grafiği
plt.figure(figsize=(10, 6))
sns.barplot(x=list(cv_results.keys()), y=list(cv_results.values()))
plt.title('Cross-Validation R² Skoru')
plt.ylabel('R² Skoru')
plt.xlabel('Modeller')
plt.xticks(rotation=45)
plt.show()
