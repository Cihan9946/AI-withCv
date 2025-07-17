import pandas as pd

# CSV dosya yolları (bunları kendi dosya yollarınla değiştir)
csv1 = "1.csv"
csv2 = "scrapped_results.csv"
csv3 = "cv_ocr_sonuclari.csv"

# Her bir CSV dosyasını oku
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df3 = pd.read_csv(csv3)

# Dosyaları birleştir
birlesik_df = pd.concat([df1, df2, df3], ignore_index=True)

# Sonuçları incele (ilk 5 satırı göster)
print(birlesik_df.head())

# Gerekirse yeni bir CSV dosyasına kaydet
birlesik_df.to_csv("merge.csv", index=False)
