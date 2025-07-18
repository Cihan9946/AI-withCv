import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# CSV dosyasını oku
df = pd.read_csv("C:\\Users\\incir\\Desktop\\cvAI\\final_Dataset.csv")

# Eğer "score" sütunu yoksa örnek amaçlı rastgele skor ekle
if 'score' not in df.columns:
    np.random.seed(42)
    df['score'] = np.random.uniform(30, 95, size=len(df))  # Gerçek veride bu satırı kaldır

# BERT model ve tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# GPU varsa kullan
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Metni BERT ile vektörleştirme fonksiyonu
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Tüm CV'leri vektörleştir
text_embeddings = []
for text in tqdm(df['text'], desc="BERT vektörleştirme"):
    try:
        embedding = get_bert_embedding(str(text))
    except:
        embedding = np.zeros(768)  # Hatalı satır varsa sıfır vektör ata
    text_embeddings.append(embedding)
text_embeddings = np.array(text_embeddings)

# Sayısal sütunlar
numeric_features = [
    'num_jobs', 'num_educations', 'num_skills', 'num_certifications',
    'years_experience', 'num_languages', 'num_projects', 'has_contact_info'
]
X_numeric = df[numeric_features].fillna(0).astype(float).values
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Özellikleri birleştir
X = np.hstack([text_embeddings, X_numeric_scaled])
y = df['score'].values

# Veriyi eğitim/test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge regresyon modeli eğit
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)

# Test verisi ile tahmin yap ve performansı ölç
y_pred = model_ridge.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Tüm veri için score tahmini
df['score'] = model_ridge.predict(X)
df['score'] = df['score'].clip(0, 100)  # Skorları 0-100 aralığında tut

# Sonuçları kaydet
df.to_csv("scored_cvs.csv", index=False)
print(df[['category', 'score']].head())
