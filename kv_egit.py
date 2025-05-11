import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


df = pd.read_csv("kahve_verisi.csv")  # Dosya adını gerektiği gibi değiştirin

# Özellikler ve hedef değişkeni belirle
X = df.drop(columns=["Coffee_Consumption_kg", "Country"])  # Ülkeyi çıkardık çünkü modelde kullanılmayacak
y = df["Coffee_Consumption_kg"]

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Test verileri ile tahmin yap
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE (Hata): {rmse:.2f}")
print(f"R² (Açıklanan Varyans): {r2:.2f}")
