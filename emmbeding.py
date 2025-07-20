
# Gerekli kütüphaneler
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# CSV dosyasını oku
df = pd.read_csv(r"C:\Users\pc\Desktop\AI-withCv-main\final_Dataset.csv")

# text sütunundaki boşlukları temizle
df["text"] = df["text"].fillna("")

# İlk 100 CV ile çalış (isteğe göre arttırabilirsin)
texts = df["text"].tolist()

# Rule-based puanlama fonksiyonu
def rule_based_score(text):
    text = str(text).lower()
    score = 50  # Başlangıç puanı

    # Programlama dilleri
    if "python" in text: score += 10
    if "java" in text: score += 7
    if "javascript" in text or "js" in text: score += 7
    if "c++" in text: score += 7
    if "c#" in text: score += 6
    if "typescript" in text: score += 5

    # Backend teknolojileri
    if "django" in text: score += 5
    if "flask" in text: score += 5
    if "node.js" in text or "express" in text: score += 5
    if "spring boot" in text: score += 5

    # Frontend teknolojileri
    if "react" in text: score += 5
    if "angular" in text: score += 5
    if "vue" in text: score += 5
    if "html" in text and "css" in text: score += 3

    # Veritabanları
    if "mysql" in text or "postgresql" in text or "sqlite" in text: score += 5
    if "mongodb" in text or "nosql" in text: score += 4

    # Bulut ve devops
    if "aws" in text: score += 6
    if "azure" in text: score += 4
    if "gcp" in text: score += 4
    if "docker" in text: score += 5
    if "kubernetes" in text: score += 5
    if "ci/cd" in text or "jenkins" in text: score += 4

    # Diğer
    if "git" in text or "version control" in text: score += 4
    if "unit testing" in text or "pytest" in text or "junit" in text: score += 3
    if "oop" in text or "object oriented" in text: score += 3
    if "rest api" in text or "graphql" in text: score += 4
    if "agile" in text or "scrum" in text: score += 2

    # Eğitim
    if "bachelor" in text or "b.sc" in text: score += 5
    if "master" in text or "m.sc" in text: score += 10
    if "phd" in text: score += 12
    if "university" in text: score += 4

    # Deneyim
    if "experience" in text:
        import re
        match = re.search(r"(\d+)\s+years? of experience", text)
        if match:
            years = int(match.group(1))
            score += min(years * 2, 10)
        else:
            score += 5  # Belirtildiyse ama sayı yoksa

    return min(score, 100)


# Her CV'ye puan ver
df["score"] = df["text"].apply(rule_based_score)
scores = df["score"].tolist()

# BERT modelini yükle
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Embedding çıkar
embeddings = []
with torch.no_grad():
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embeddings.append(cls_vector)

# Eğitim ve test veri setini ayır
X_train, X_test, y_train, y_test = train_test_split(embeddings, scores, test_size=0.2, random_state=42)

# Ridge regresyon modelini eğit
reg_model = Ridge()
reg_model.fit(X_train, y_train)

# Tahmin ve değerlendirme
y_pred = reg_model.predict(X_test)
print("R² skoru:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Modeli diske kaydet
joblib.dump(reg_model, "cv_score_model_new.pkl")
print("✅ Model başarıyla 'cv_score_model.pkl' dosyasına kaydedildi.")
