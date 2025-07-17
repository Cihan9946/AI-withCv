import pdfplumber
import joblib

# PDF'ten metni çıkar
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# PDF yolunu verin
pdf_path = "skills-based-cv.pdf"  # Buraya kendi PDF dosyanızın adını yazın

# 1. PDF'i oku
cv_text = extract_text_from_pdf(pdf_path)

# 2. Vektörizer ve modeli yükle
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/logistic_regression_model.pkl")

# 3. Metni vektöre çevir ve tahmin et
vectorized_cv = vectorizer.transform([cv_text])
prediction = model.predict(vectorized_cv)

# 4. Sonucu yazdır
print(f" Tahmin edilen kategori: {prediction[0]}")
