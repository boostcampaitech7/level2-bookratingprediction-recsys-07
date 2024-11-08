# main.py

import argparse
from preprocess.wnd_data import load_data
from train.wnd_train import train_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()

    # 데이터 로드 및 전처리
    train_dataset, valid_dataset, test_dataset, field_dims = load_data(args)

    # 모델 학습 및 검증
    model = train_model(train_dataset, valid_dataset, field_dims, args)

if __name__ == "__main__":
    main()
