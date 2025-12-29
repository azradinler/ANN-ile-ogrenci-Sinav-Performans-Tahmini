# ==========================================
# Student Performance Regression with ANN
# ==========================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------
# 1. Veri Setini Yükle
# -------------------------------
df = pd.read_csv("Student_Performance.csv")

# 🔴 Sütun isimlerini temizle
df.columns = df.columns.str.replace(" ", "_")

print("İlk 5 Satır:")
print(df.head())

# -------------------------------
# 2. Kategorik Veriyi Encode Et
# -------------------------------
df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({
    "Yes": 1,
    "No": 0
})

# -------------------------------
# 3. Girdi (X) ve Çıktı (y)
# -------------------------------
X = df.drop("Performance_Index", axis=1)
y = df["Performance_Index"]

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 5. Train / Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Tensor Dönüşümü
# -------------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# -------------------------------
# 7. DataLoader
# -------------------------------
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# -------------------------------
# 8. ANN Model
# -------------------------------
class StudentANN(nn.Module):
    def __init__(self, input_size):
        super(StudentANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = StudentANN(X_train.shape[1])

# -------------------------------
# 9. Eğitim Ayarları
# -------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 200

# -------------------------------
# 10. Model Eğitimi
# -------------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

# -------------------------------
# 11. Model Değerlendirme
# -------------------------------
model.eval()
with torch.no_grad():
    predictions = model(X_test)

y_true = y_test.numpy()
y_pred = predictions.numpy()

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\nModel Performansı:")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.2f}")

# -------------------------------
# 12. Örnek Tahminler
# -------------------------------
print("\nİlk 5 Gerçek vs Tahmin:")
for i in range(15):
    print(f"Gerçek: {y_true[i][0]:.2f} | Tahmin: {y_pred[i][0]:.2f}")

import gradio as gr

# -------------------------------
# 13. Tahmin Fonksiyonu
# -------------------------------
def predict_performance(
    hours_studied,
    previous_scores,
    extracurricular,
    sleep_hours,
    sample_papers
):
    input_data = np.array([[
        hours_studied,
        previous_scores,
        extracurricular,
        sleep_hours,
        sample_papers
    ]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Tahmin
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).item()

    # 🔒 SINIRLAMA (0 - 100)
    prediction = max(0, min(100, prediction))

    # 🔢 İstersen yuvarla
    prediction = round(prediction, 2)

    return prediction

# -------------------------------
# 14. Gradio UI
# -------------------------------
interface = gr.Interface(
    fn=predict_performance,
    inputs=[
        gr.Number(label="Hours Studied"),
        gr.Number(label="Previous Scores"),
        gr.Radio([0, 1], label="Extracurricular Activities (0=No, 1=Yes)"),
        gr.Number(label="Sleep Hours"),
        gr.Number(label="Sample Question Papers Practiced")
    ],
    outputs=gr.Number(label="Predicted Performance Index"),
    title="🎓 Student Performance Prediction (ANN)",
    description="Bu uygulama PyTorch ile eğitilmiş Yapay Sinir Ağı kullanarak öğrenci performansını tahmin eder."
)

# -------------------------------
# 15. Uygulamayı Başlat
# -------------------------------
interface.launch()
