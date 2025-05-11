import pandas as pd
import numpy as np


df = pd.read_csv("duzenlenmis_kahve_verisi.csv")  # Kendi dosya adını kullan

# 1. Kahve türü sayısı ve karışım durumu
coffee_types = [
    "Coffee_Type_Americano",
    "Coffee_Type_Cappuccino",
    "Coffee_Type_Espresso",
    "Coffee_Type_Latte",
    "Coffee_Type_Mocha"
]

df["total_types"] = df[coffee_types].sum(axis=1)
df["is_mixed"] = (df["total_types"] > 1).astype(int)

# 2. Zaman tabanlı türetmeler
df["year_diff_squared"] = df["Year_diff"] ** 2
df["year_diff_cubed"] = df["Year_diff"] ** 3
df["yearly_trend_positive"] = (df["yearly_trend"] > 0).astype(int)

# 3. Fiyat ve nüfus etkileşimleri
df["log_price"] = np.log1p(df["standardized_price"])
df["price_per_capita_log"] = df["standardized_price"] / np.exp(df["log_population"])
df["price_trend_interaction"] = df["standardized_price"] * df["yearly_trend"]

# 4. Kahve türü bazlı etkileşimler (örnek olarak Americano)
df["americano_price_interaction"] = df["Coffee_Type_Americano"] * df["standardized_price"]
df["americano_year_trend"] = df["Coffee_Type_Americano"] * df["yearly_trend"]


df.to_csv("veri_gelistirilmis.csv", index=False)

print("Yeni özellikler eklendi ve 'veri_gelistirilmis.csv' olarak kaydedildi.")
