import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from models.wnd import WideAndDeep
from preprocess.wnd_data import load_data
import argparse

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}")

        if valid_loader is not None:
            validate_model(model, valid_loader, criterion, device)

def validate_model(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    avg_valid_loss = total_loss / len(valid_loader)
    print(f"Validation Loss: {avg_valid_loss:.4f}")

def create_submission_file(model, test_loader, submission_template, device='cpu'):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().flatten())

    submission = submission_template.copy()
    submission['rating'] = predictions
    submission.to_csv('submission.csv', index=False)
    print("submission.csv 파일이 생성되었습니다.")

if __name__ == "__main__":
    # Argument 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=8)
    parser.add_argument('--mlp_dims', type=int, nargs='+', default=[64, 32])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batchnorm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # 데이터 로드 및 전처리
    train_dataset, valid_dataset, test_dataset, field_dims = load_data(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False) if valid_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 모델 초기화
    model = WideAndDeep(args, {'field_dims': field_dims})
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 모델 학습
    train_model(model, train_loader, valid_loader, criterion, optimizer, args.device, args.epochs)

    # 테스트 데이터에 대해 예측 후 submission 파일 생성
    submission_template = pd.read_csv(args.data_path + 'sample_submission.csv')
    create_submission_file(model, test_loader, submission_template, device=args.device)
