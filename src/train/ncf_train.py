# main.py (또는 train.py)
from dataset.ncf_dataset import load_datasets  # 데이터 로드 함수
from models.ncf_models import NCF, NeuMF       # 모델 클래스
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# 하이퍼파라미터 설정
batch_size = 64
embedding_dim = 32
hidden_layers = [64, 32, 16]
dropout_rate = 0.3
learning_rate = 0.001
num_epochs = 10

# 데이터 로드
train_loader, val_loader, test_loader, predict_test_loader, num_users, num_items = load_datasets(
    path='data/', batch_size=batch_size
)

# 모델 초기화 (NCF 또는 NeuMF 선택)
model = NeuMF(num_users, num_items, embedding_dim=embedding_dim, hidden_layers=hidden_layers, dropout_rate=dropout_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수와 옵티마이저 정의
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 함수
def train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for user_id, item_id, rating in tqdm(data_loader, desc="Training"):
        user_id, item_id, rating = user_id.to(device), item_id.to(device), rating.to(device)
        
        optimizer.zero_grad()
        prediction = model(user_id, item_id)
        loss = loss_fn(prediction, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# 평가 함수
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for user_id, item_id, rating in tqdm(data_loader, desc="Evaluating"):
            user_id, item_id, rating = user_id.to(device), item_id.to(device), rating.to(device)
            prediction = model(user_id, item_id)
            loss = loss_fn(prediction, rating)
            total_loss += loss.item()
    rmse = torch.sqrt(torch.tensor(total_loss / len(data_loader)))
    return rmse.item()

# 학습 및 평가 루프
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, loss_fn, optimizer, device)
    val_rmse = evaluate(model, val_loader, loss_fn, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation RMSE: {val_rmse:.4f}")

# 테스트 데이터 예측 (옵션)
test_rmse = evaluate(model, test_loader, loss_fn, device)
print(f"Test RMSE: {test_rmse:.4f}")
