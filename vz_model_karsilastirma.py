import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ------------------- 1. VERİYİ YÜKLE -------------------
df = pd.read_csv("veri_zenginlestirilmis.csv")

# Hedef değişkeni ayarla (gerekirse değiştir)
target_col = "Coffee_Consumption_kg"

X = df.drop(columns=[target_col])
y = df[target_col]

# ------------------- 2. VERİYİ BÖL -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------- 3. MODELLER -------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR(),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
}

# ------------------- 4. EĞİTİM VE DEĞERLENDİRME -------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    results.append({
        "Model": name,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse
    })

# ------------------- 5. RAPORU OLUŞTUR -------------------
results_df = pd.DataFrame(results)
print("\n🔍 Model Karşılaştırma Raporu:\n")
print(results_df.sort_values(by="Test R2", ascending=False))

# ------------------- 6. GÖRSEL ANALİZ -------------------
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df.melt(id_vars="Model", value_vars=["Train RMSE", "Test RMSE"]),
            x="Model", y="value", hue="variable")
plt.title("RMSE Karşılaştırması (Daha düşük daha iyi)")
plt.ylabel("RMSE")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("rmse_karsilastirma.png")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df.melt(id_vars="Model", value_vars=["Train R2", "Test R2"]),
            x="Model", y="value", hue="variable")
plt.title("R² Karşılaştırması (Daha yüksek daha iyi)")
plt.ylabel("R² Skoru")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("r2_karsilastirma.png")
plt.show()
