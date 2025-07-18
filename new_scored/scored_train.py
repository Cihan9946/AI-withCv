import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import joblib

# 1. Veri yükleme
df = pd.read_csv("scored_cvs.csv")

# 2. Giriş ve hedef sütunları ayır
excluded_cols = ['text', 'score', 'category']
numerical_features = df.drop(columns=excluded_cols, errors='ignore')
text_data = df['text'].fillna("")
target = df['score'].values

# 3. Sayısal sütunları standardize et
scaler = StandardScaler()
X_numeric = scaler.fit_transform(numerical_features)

# 4. BERT tokenizer ve model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()  # fine-tune etmeyeceğiz

# 5. BERT ile metin vektörleştirme
def get_bert_embedding(text_list):
    embeddings = []
    with torch.no_grad():
        for text in tqdm(text_list, desc="BERT embedding"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
    return np.array(embeddings)

X_text = get_bert_embedding(text_data)

# 6. BERT + Sayısal birleşimi
X_combined = np.concatenate([X_text, X_numeric], axis=1)

# 7. Train-test bölme
X_train, X_test, y_train, y_test = train_test_split(X_combined, target, test_size=0.2, random_state=42)

# 8. Dataset
class CVBertDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CVBertDataset(X_train, y_train)
test_dataset = CVBertDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 9. PyTorch Regresyon Modeli
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

model = CVBertRegressor(input_dim=X_combined.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 10. Eğitim
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

# 11. Test Değerlendirmesi
model.eval()
preds = []
actuals = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        output = model(batch_x)
        preds.extend(output.view(-1).tolist())
        actuals.extend(batch_y.view(-1).tolist())

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(actuals, preds)
r2 = r2_score(actuals, preds)
print(f"Test MSE: {mse:.2f}")
print(f"Test R²: {r2:.2f}")

# 12. Model ve scaler kaydetme
torch.save(model.state_dict(), "bert_cv_score_model.pt")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model ve scaler başarıyla kaydedildi.")
