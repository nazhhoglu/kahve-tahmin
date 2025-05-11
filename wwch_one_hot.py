import pandas as pd

# 1. Veriyi oku
df = pd.read_csv("worldwide_coffee_habits.csv")  # kendi dosya adına göre değiştir

# 2. Kahve türlerini One-Hot Encoding'e çevir
df_encoded = pd.get_dummies(df, columns=["Coffee_Type"])

# 3. Sonuçları kontrol et (ilk 5 satır)
print("Yeni sütunlar:", df_encoded.columns)
print("İlk 5 satır:\n", df_encoded.head())

# 4. Yeni veriyi CSV olarak kaydet
df_encoded.to_csv("kahve_verisi.csv", index=False)
print("Encoded veri başarıyla 'kahve_verisi.csv' olarak kaydedildi.")
