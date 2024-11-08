import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(args):
    train_df = pd.read_csv(args.data_path + 'train_ratings.csv')
    test_df = pd.read_csv(args.data_path + 'test_ratings.csv')

    # 범주형 데이터 인코딩
    categorical_cols = ['user_id', 'isbn']
    for col in categorical_cols:
        train_df[col] = train_df[col].astype('category').cat.codes
        test_df[col] = test_df[col].astype('category').cat.codes

    field_dims = [train_df[col].nunique() for col in categorical_cols]

    # Train-validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_df[categorical_cols].values,
        train_df['rating'].values,
        test_size=args.valid_ratio,
        random_state=args.seed,
        shuffle=True
    )

    # TensorDataset 생성
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float))
    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.long), torch.tensor(y_valid, dtype=torch.float))
    test_dataset = TensorDataset(torch.tensor(test_df[categorical_cols].values, dtype=torch.long))

    return train_dataset, valid_dataset, test_dataset, field_dims
