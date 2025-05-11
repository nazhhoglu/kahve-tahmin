import pandas as pd
import numpy as np


df = pd.read_csv("veri_temizlenmis.csv")

# A. Yıl Bilgisini Açık Şekilde Kullan (varsayılan yıl_diff varsa)
df['year_since_2000'] = df['Year_diff'] + 2000  # Örneğin Year_diff = 5 ise 2005

# B. Kahve Türlerine Ait Fiyat-Oranı Etkileşimleri
kahve_turleri = ['Coffee_Type_Americano', 'Coffee_Type_Cappuccino', 
                 'Coffee_Type_Espresso', 'Coffee_Type_Latte', 'Coffee_Type_Mocha']

for tur in kahve_turleri:
    tur_adi = tur.replace('Coffee_Type_', '').lower()
    df[f'{tur_adi}_price_interaction'] = df[tur] * df['standardized_price']

# C. Coğrafi / Kültürel Gruplar (Varsa yoksa örnek amaçlı dummy sütun ekleyebiliriz)
# Örnek: df = pd.get_dummies(df, columns=['region'], drop_first=True)

# D. Tüketim Yoğunluğu (Kişi başı tüketim)
# Not: population bilgisi log_population'dan türetilebilir
df['population'] = np.expm1(df['log_population'])  # log1p uygulanmışsa tersini al
df['consumption_per_capita'] = df['Coffee_Consumption_kg'] / df['population']

# E. Çeşitlilik Skoru (Normalize edilmiş çeşit sayısı)
df['diversity_score'] = df['total_types'] / len(kahve_turleri)  # max 5 tür varsa 5'e böldük

# Yeni veri kümesini kaydet
df.to_csv("veri_zenginlestirilmis.csv", index=False)
print("Yeni özelliklerle veri seti 'veri_zenginlestirilmis.csv' olarak kaydedildi.")
