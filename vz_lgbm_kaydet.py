import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# 1. CSV'den veriyi yükle
df = pd.read_csv("veri_zenginlestirilmis.csv")

# 2. Özellik ve hedef sütunlarını ayır
X = df.drop("Coffee_Consumption_kg", axis=1)  # Hedef sütun adı
y = df["Coffee_Consumption_kg"]

# 3. Eğitim ve test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modeli oluştur ve eğit
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)

# 5. Kaydet
dump(lgb_model, "lightgbm_model.pkl")
dump(X.columns.tolist(), "model_features.pkl")
print("✅ LightGBM modeli kaydedildi: lightgbm_model.pkl")

# 6. Geri yükle
loaded_model = load("lightgbm_model.pkl")
print("📦 LightGBM modeli yüklendi ve kullanılmaya hazır.")

# 7. Tahmin ve değerlendirme
predictions = loaded_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"🎯 LightGBM Test MSE: {mse:.4f}")
