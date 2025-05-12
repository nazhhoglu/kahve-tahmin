import pandas as pd
from joblib import load

# Modeli ve özellik isimlerini yükle
model = load("lightgbm_model.pkl")
feature_names = load("model_features.pkl")

# Yeni veriyi oku
yeni_veri = pd.read_csv("veri_zenginlestirilmis.csv")

# Gerçek değerleri ayrı bir değişkende tut
gercek_degerler = yeni_veri["Coffee_Consumption_kg"]  # Hedef sütun adı

# Eksik sütunları ekle, fazlaları çıkar
for col in feature_names:
    if col not in yeni_veri.columns:
        yeni_veri[col] = 0
yeni_veri = yeni_veri[feature_names]  # Sıralama da eşleşsin

# Tahmin yap
tahminler = model.predict(yeni_veri)

# Tahmin ve gerçek değerleri birlikte yazdır
sonuc_df = pd.DataFrame({
    "Gerçek Değer": gercek_degerler,
    "Tahmin": tahminler
})

print(sonuc_df.head(10))  # İlk 10 satırı yazdır
sonuc_df.to_csv("lgbm_tahmin_vs_gercek.csv", index=False)  # CSV'ye yaz
