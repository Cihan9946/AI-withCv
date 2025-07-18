import fitz  # PyMuPDF
import pandas as pd
import re
import torch
import numpy as np
import joblib
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# -------------------------------
# 1. PDF'ten metni al ve özellikleri çıkar
# -------------------------------
pdf_path = r"C:\Users\incir\Desktop\cvAI\Mustafa Cihan İncir CV.pdf"
doc = fitz.open(pdf_path)
full_text = ""
for page in doc:
    full_text += page.get_text() + "\n"
doc.close()

def count_job_experiences(text):
    headers = ['work experience', 'experience', 'employment', 'professional experience']
    count = 0
    lines = text.lower().split('\n')
    for i, line in enumerate(lines):
        if any(header in line for header in headers):
            for j in range(i+1, min(i+10, len(lines))):
                if re.search(r'\b\d{4}\b', lines[j]):
                    count += 1
    return count

def count_educations(text):
    matches = re.findall(r'\b(university|college|b\.a|b\.sc|m\.a|m\.sc|ph\.d|education)\b', text.lower())
    return len(matches)

def count_skills(text):
    skill_section = re.search(r'(skills|technical skills)(.*?)\n\n', text.lower(), re.DOTALL)
    if skill_section:
        content = skill_section.group(2)
        return len(re.split(r',|\n', content.strip()))
    return 0

def count_certifications(text):
    matches = re.findall(r'\b(certification|certifications|certificate|certified)\b', text.lower())
    return len(matches)

def estimate_experience_years(text):
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    years = list(map(int, years))
    if len(years) >= 2:
        return max(years) - min(years)
    return 0

def count_languages(text):
    match = re.search(r'(languages|fluent in)(.*)', text.lower())
    if match:
        line = match.group(2)
        return len(re.findall(r'[a-zA-Z]+', line))
    return 0

def count_projects(text):
    matches = re.findall(r'\b(project|projects|project management|project experience)\b', text.lower())
    return len(matches)

def has_contact_info(text):
    email = re.search(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', text)
    phone = re.search(r'(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', text)
    address = re.search(r'\d{1,5} [\w\s]+,? [\w\s]+', text)
    return int(bool(email or phone or address))

# Özellikleri çıkar
data = {
    "text": full_text,
    "num_jobs": count_job_experiences(full_text),
    "num_educations": count_educations(full_text),
    "num_skills": count_skills(full_text),
    "num_certifications": count_certifications(full_text),
    "years_experience": estimate_experience_years(full_text),
    "num_languages": count_languages(full_text),
    "num_projects": count_projects(full_text),
    "has_contact_info": has_contact_info(full_text)
}
df = pd.DataFrame([data])

# -------------------------------
# 2. Modeli yükle ve tahmin yap
# -------------------------------

# Tokenizer ve BERT yükle
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# Scaler yükle
scaler = joblib.load("scaler.pkl")

# Metni vektörleştir
def get_bert_embedding(text_list):
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
    return np.array(embeddings)

X_text = get_bert_embedding(df['text'].tolist())

# Sayısal sütunları hazırlama
numerical_features = df.drop(columns=["text", "category", "score"], errors='ignore')

# Eksik olan feature'ları scaler'a göre sıfırla (örn: Unnamed: 0)
for col in scaler.feature_names_in_:
    if col not in numerical_features.columns:
        numerical_features[col] = 0

# Sıralamayı scaler.feature_names_in_ ile eşleştir
numerical_features = numerical_features[scaler.feature_names_in_]

# Ölçekle
X_numeric = scaler.transform(numerical_features)

# Vektörleri birleştir
X_combined = np.concatenate([X_text, X_numeric], axis=1)
X_tensor = torch.tensor(X_combined, dtype=torch.float32)

# Model sınıfı
class CVBertRegressor(nn.Module):
    def __init__(self, input_dim):
        super(CVBertRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# Eğitilmiş modeli yükle
model = CVBertRegressor(input_dim=X_tensor.shape[1])
model.load_state_dict(torch.load("bert_cv_score_model.pt"))
model.eval()

# Tahmin
with torch.no_grad():
    prediction = model(X_tensor).item()

print(f"\n✅ PDF CV için tahmini skor: {prediction:.2f} / 100")
