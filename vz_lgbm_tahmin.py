import pandas as pd
from joblib import load
from sklearn.metrics import r2_score, mean_absolute_percentage_error

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

# Doğruluk metrikleri
r2 = r2_score(gercek_degerler, tahminler)
mape = mean_absolute_percentage_error(gercek_degerler, tahminler) * 100  # % cinsinden

# Tahmin ve gerçek değerleri birlikte yazdır
sonuc_df = pd.DataFrame({
    "Gerçek Değer": gercek_degerler,
    "Tahmin": tahminler
})

print("İlk 10 sonuç:")
print(sonuc_df.head(10))
print(f"\nModel R² skoru: {r2:.4f}")
print(f"Ortalama yüzde hata (MAPE): %{mape:.2f}")
print(f"Yaklaşık doğruluk oranı: %{100 - mape:.2f}")


sonuc_df.to_csv("lgbm_tahmin_vs_gercek.csv", index=False)  # CSV'ye yaz
