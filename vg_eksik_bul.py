import pandas as pd
import numpy as np


dosya_adi = "veri_gelistirilmis.csv"
df = pd.read_csv(dosya_adi)

# Eksik değerleri özetle
print("Eksik Değerler (Önce):")
print(df.isnull().sum())

# Sayısal sütunları ortalama ile doldur
sayisal_sutunlar = df.select_dtypes(include=[np.number]).columns
for sutun in sayisal_sutunlar:
    df[sutun] = df[sutun].fillna(df[sutun].mean())

# Kategorik sütunları "Unknown" ile doldur
kategorik_sutunlar = df.select_dtypes(include=["object"]).columns
for sutun in kategorik_sutunlar:
    df[sutun] = df[sutun].fillna("Unknown")

# Eksik değerler düzeltildikten sonra kontrol et
print("\nEksik Değerler (Sonra):")
print(df.isnull().sum())


df.to_csv("veri_temizlenmis.csv", index=False)
print("\nTemizlenmiş veri 'veri_temizlenmis.csv' olarak kaydedildi.")
