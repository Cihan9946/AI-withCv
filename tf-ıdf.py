import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. CSV dosyasını oku
df = pd.read_csv("merge.csv")

# 2. Eksik verileri temizle
df = df.dropna(subset=['text', 'category'])

# 3. TF-IDF vektörleştirme
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['text'])
y = df['category']

# 4. Eğitim/test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Logistic Regression modeli eğit
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Performans çıktısı
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Modelleri kaydet
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(model, "models/logistic_regression_model.pkl")

print("✅ Model ve vectorizer 'models/' klasörüne kaydedildi.")
