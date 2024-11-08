# src/train/train_wnd.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.wnd import WideAndDeep

def train_model(train_dataset, valid_dataset, field_dims, args):
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # 모델 초기화
    model = WideAndDeep(field_dims, args.embed_dim).to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 모델 학습
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        validate_model(model, valid_loader, criterion, args.device)

    return model

def validate_model(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss/len(valid_loader):.4f}")
