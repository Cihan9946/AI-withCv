import os
import csv
from PIL import Image
import pytesseract

# Eğer Windows'ta isen Tesseract motorunun yolunu belirt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# OCR yapılacak ana klasörün yolu
ana_klasor = 'Scrapped_Resumes'  # <-- burayı kendi klasör yolunla değiştir

# CSV dosyasını oluştur
csv_dosya_adi = 'scrapped_results.csv'

# Geçerli uzantılar
gecerli_uzantilar = ['.png', '.jpg', '.jpeg']

# CSV dosyasını yazma modunda aç
with open(csv_dosya_adi, mode='w', newline='', encoding='utf-8') as csv_dosyasi:
    writer = csv.writer(csv_dosyasi)
    writer.writerow(['text', 'category'])  # Başlıkları yaz

    # Alt klasörlerde gezin
    for kategori in os.listdir(ana_klasor):
        kategori_yolu = os.path.join(ana_klasor, kategori)

        # Eğer bu bir klasörse
        if os.path.isdir(kategori_yolu):
            # Klasör içindeki dosyalarda gez
            for dosya_adi in os.listdir(kategori_yolu):
                tam_dosya_yolu = os.path.join(kategori_yolu, dosya_adi)

                # Geçerli resim uzantılarına sahip mi?
                if any(dosya_adi.lower().endswith(uzanti) for uzanti in gecerli_uzantilar):
                    try:
                        # Görseli aç ve OCR yap
                        img = Image.open(tam_dosya_yolu)
                        metin = pytesseract.image_to_string(img, lang='eng')  # Türkçe için lang='tur'

                        # Satırı CSV'ye yaz
                        writer.writerow([metin.strip(), kategori])
                        print(f"✅ {dosya_adi} -> {kategori} kategorisine yazıldı.")

                    except Exception as e:
                        print(f"⚠️ Hata ({dosya_adi}): {e}")

print(f"\n✅ Tüm veriler '{csv_dosya_adi}' adlı dosyaya kaydedildi.")