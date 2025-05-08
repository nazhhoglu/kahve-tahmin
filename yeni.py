import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Veri Yükleme
# CSV dosyanızı bu dosyayla aynı klasöre yerleştirin.
df = pd.read_csv("duzenlenmis_kahve_verisi.csv")

# Veri Ön İnceleme
df.head()

# Temel İstatistikler
df.describe()

# Eksik Veri Kontrolü
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Eksik Veriler Isı Haritası")
plt.show()

# Korelasyon Matrisi
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()

# Hedef Değişken Dağılımı
plt.figure(figsize=(8, 5))
sns.histplot(df["Coffee_Consumption_kg"], kde=True, bins=30)
plt.title("Kahve Tüketimi Dağılımı")
plt.xlabel("Coffee_Consumption_kg")
plt.ylabel("Frekans")
plt.show()

# Özellik ve Etiket Ayırma
X = df.drop("Coffee_Consumption_kg", axis=1)
y = df["Coffee_Consumption_kg"]

# Eğitim ve Test Seti
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Eğitimi
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)

# Değerlendirme Metrikleri
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Skoru:", r2)
print("Model Doğruluğu (%):", r2 * 100)

# Tahmin vs Gerçek
sonuclar = pd.DataFrame({"Gerçek": y_test, "Tahmin": y_pred})
sonuclar["Hata"] = sonuclar["Gerçek"] - sonuclar["Tahmin"]
sonuclar.head()

# Grafik: Gerçek vs Tahmin
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Gerçek", marker='o')
plt.plot(y_pred, label="Tahmin", linestyle='dashed', marker='x')
plt.title("Gerçek vs Tahmin Edilen Kahve Tüketimi")
plt.xlabel("Gözlem")
plt.ylabel("Coffee_Consumption_kg")
plt.legend()
plt.grid(True)
plt.show()

# Özellik Katsayıları
coef_df = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
coef_df.plot(kind="bar")
plt.title("Özelliklerin Model Üzerindeki Etkisi (Katsayılar)")
plt.ylabel("Katsayı Değeri")
plt.grid(True)
plt.show()
