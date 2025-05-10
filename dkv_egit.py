import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Veri Yükleme
df = pd.read_csv("duzenlenmis_kahve_verisi.csv")

# İlk Bakış
print(df.head())
print(df.describe())

# Eksik Veri Görselleştirme
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Eksik Veriler Isı Haritası")
plt.show()

# Aykırı Değer Analizi
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["Coffee_Consumption_kg"])
plt.title("Kahve Tüketiminde Aykırı Değerler")
plt.show()

# Korelasyon Matrisi
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()

# Yeni Özellikler
population = np.exp(df["log_population"])
df["price_per_capita"] = df["standardized_price"] / population
df["year_diff_squared"] = df["Year_diff"] ** 2
df["year_diff_log"] = np.log1p(df["Year_diff"])

# Özellik ve Hedef Ayırma
X = df.drop("Coffee_Consumption_kg", axis=1)
y = df["Coffee_Consumption_kg"]

# Özellik Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim-Test Ayırımı
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modeller
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Performans Değerlendirme
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "Accuracy (%)": r2 * 100
    })

# Sonuç Tablosu
result_df = pd.DataFrame(results)

print(result_df.sort_values(by="R2", ascending=False))


# En iyi modelin tahmin vs gerçek karşılaştırması
best_model_name = result_df.sort_values(by="R2", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Gerçek", marker='o')
plt.plot(y_pred_best, label="Tahmin", linestyle='dashed', marker='x')
plt.title(f"{best_model_name} - Gerçek vs Tahmin Edilen Kahve Tüketimi")
plt.xlabel("Gözlem")
plt.ylabel("Coffee_Consumption_kg")
plt.legend()
plt.grid(True)
plt.show()

# En iyi modelin özellik katsayıları (varsa)
if hasattr(best_model, 'coef_'):
    coef_df = pd.Series(best_model.coef_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    coef_df.plot(kind="bar")
    plt.title(f"{best_model_name} Modeli Özellik Katsayıları")
    plt.ylabel("Katsayı Değeri")
    plt.grid(True)
    plt.show()

# Cross Validation ile en iyi model
cv_score = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
print(f"{best_model_name} için Ortalama R2 (CV):", cv_score.mean())
