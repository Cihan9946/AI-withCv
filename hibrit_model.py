import pdfplumber
import joblib
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# 🔹 1. PDF'ten metni çıkar
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

# PDF yolu
pdf_path = r"C:\Users\incir\Desktop\scored_new\Mustafa Cihan İncir CV.pdf"

cv_text = extract_text_from_pdf(pdf_path)

# 🔹 2. Kategori tahmini (TF-IDF + Logistic Regression)

vectorizer = joblib.load("C:\\Users\\incir\\Desktop\\scored_new\\models\\tfidf_vectorizer.pkl")
model_cls = joblib.load("C:\\Users\\incir\\Desktop\\scored_new\\models\\logistic_regression_model.pkl")

vectorized_cv = vectorizer.transform([cv_text])
predicted_category = model_cls.predict(vectorized_cv)[0]

# 🔹 3. CV puan tahmini (BERT + Ridge Regressor)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)
bert_model.eval()

# Metni embed et (BERT)
with torch.no_grad():
    inputs = tokenizer(cv_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Skor modelini yükle
reg_model = joblib.load("cv_score_model_new.pkl")

# Tahmini skor
predicted_score = reg_model.predict([cls_embedding])[0]

# 🔹 4. Sonuçları yazdır
print(f"📂 Tahmin edilen kategori: {predicted_category}")
print(f"⭐ CV Skoru (0–100): {predicted_score:.2f}")
