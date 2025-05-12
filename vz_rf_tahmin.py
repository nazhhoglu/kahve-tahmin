import pandas as pd
from joblib import load

# Modeli ve özellik isimlerini yükle
model = load("random_forest_model.pkl")
feature_names = load("model_features.pkl")

# Yeni veriyi oku
yeni_veri = pd.read_csv("veri_zenginlestirilmis.csv")

# Eksik sütunları ekle, fazlaları çıkar
for col in feature_names:
    if col not in yeni_veri.columns:
        yeni_veri[col] = 0  # eksik olan sütunları 0 olarak ekle
yeni_veri = yeni_veri[feature_names]  # sıralamayı da eşleştir

# Tahmin yap
tahminler = model.predict(yeni_veri)
print(tahminler)
