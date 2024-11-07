import argparse
import torch
import random
import os
import numpy as np
import torch.nn as nn
from torch.nn import MSELoss
import ast
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import sys

try:
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
except Exception as e:
    print(e)
    sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

from preprocess.DCN_dataset import pre_context_data
from models.DCN import DeepCrossNetwork

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_DCN(args):
    seed_everything(args.seed)
    if args.device == 'cuda' and torch.cuda.is_available():
        print("cuda use success")
    
    print(f'-------------- LOAD DATA ---------------')
    data = pre_context_data(args.filepath)
    
    print(f'-------------- Train/Valid Split ---------------')
    if args.ratio > 0:
        X_train, X_valid, y_train, y_valid = train_test_split(
            data['train'].drop(['rating'], axis=1),
            data['train']['rating'],
            test_size=args.ratio,
            random_state=args.seed,
            shuffle=True
        )
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    else:
        data['X_train'], data['y_train'] = data['train'].drop(['rating'], axis=1), data['train']['rating']
        data['X_valid'], data['y_valid'] = None, None

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values)) if args.ratio != 0 else None
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if args.ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f'--------------- INIT model ---------------')
    
    model = DeepCrossNetwork(args, data).to(args.device)
    criterion = RMSELoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in tqdm(train_dataloader, desc=f'[Epoch {epoch+1}/{args.epochs}]'):
            X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.float())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_train_loss / len(train_dataloader.dataset)

        if valid_dataloader:
            model.eval()
            total_valid_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in valid_dataloader:
                    X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch.float())
                    total_valid_loss += loss.item() * X_batch.size(0)

            avg_valid_loss = total_valid_loss / len(valid_dataloader.dataset)
            print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}')

    print('--------------- FINAL TRAINING AND PREDICTION ---------------')
    model = DeepCrossNetwork(args, data).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    train_dataset_full = TensorDataset(torch.LongTensor(data['train'].drop(['rating'], axis=1).values), torch.LongTensor(data['train']['rating'].values))
    train_dataloader_full = DataLoader(train_dataset_full, batch_size=args.batch_size, shuffle=True, num_workers=0)

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_dataloader_full:
            X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_train_loss / len(train_dataloader_full.dataset)
        print(f'Final Training Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}')

    predictions = []
    with torch.no_grad():
        for X_batch in test_dataloader:
            X_batch = X_batch[0].to(args.device)
            y_pred = model(X_batch)
            predictions.extend(y_pred.cpu().numpy().tolist())

    data['sub']['rating'] = predictions
    data['sub'].to_csv('./submission.csv', index=False)
    print(f'Submission file saved')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PARSER')
    arg = parser.add_argument
    
    parser.add_argument('--filepath', '-fp', type=str, help='File path를 지정합니다')
    parser.add_argument('--seed', type=int, help='초기 사용 시드를 고정합니다')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='사용할 디바이스를 선택가능')
    parser.add_argument('--ratio', type=float, help='train valid dataset ratio')
    parser.add_argument('--lr', type=float, help='learning rate 지정')
    parser.add_argument('--cross_layer_num', type=int, help='cross layer의 개수 설정')
    parser.add_argument('--mlp_dims', type=list, help='the list of mlp_dims')
    parser.add_argument('--embed_dim', type=int, help='embedding dimension')
    parser.add_argument('--batchnorm', type=ast.literal_eval, help='batchnorm 설정 (True 또는 False)')
    parser.add_argument('--dropout', type=float, help='dropout 비율 설정')
    parser.add_argument('--batch_size', type=int, help='batchsize 설정')
    parser.add_argument('--epochs', type=int, help='epochs 정하기')
    
    args = parser.parse_args()
    
    args.filepath = args.filepath if args.filepath is not None else "/data/ephemeral/home/lee/level2-bookratingprediction-recsys-07/data"
    args.seed = args.seed if args.seed is not None else 42
    args.device = args.device if args.device is not None else 'cuda'
    args.ratio = args.ratio if args.ratio is not None else 0.2
    args.lr = args.lr if args.lr is not None else 0.005
    args.cross_layer_num = args.cross_layer_num if args.cross_layer_num is not None else 3
    args.mlp_dims = args.mlp_dims if args.mlp_dims is not None else [16, 32]
    args.embed_dim = args.embed_dim if args.embed_dim is not None else 16
    args.batchnorm = args.batchnorm if args.batchnorm is not None else True
    args.dropout = args.dropout if args.dropout is not None else 0.2
    args.batch_size = args.batch_size if args.batch_size is not None else 128
    args.epochs = args.epochs if args.epochs is not None else 30
    
    print(f"args: {args}")
    
    train_DCN(args)