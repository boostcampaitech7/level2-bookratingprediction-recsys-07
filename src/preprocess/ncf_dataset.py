import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class RatingsDataset(Dataset):
    def __init__(self, data):
        self.user_ids = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.item_ids = torch.tensor(data['isbn'].values, dtype=torch.long)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float) if 'rating' in data.columns else None

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        if self.ratings is not None:
            return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]
        else:
            return self.user_ids[idx], self.item_ids[idx]

def load_datasets(path='data/', train_file='train_ratings.csv', test_file='test_ratings.csv', 
                  batch_size=64, shuffle_train=True, shuffle_val=False):
    # 데이터 로드
    train_ratings = pd.read_csv(path + train_file)
    test_ratings = pd.read_csv(path + test_file)

    # 사용자와 아이템 ID를 정수 인덱스로 변환
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    # 훈련 데이터로 인코더를 학습
    train_ratings['user_id'] = user_encoder.fit_transform(train_ratings['user_id'])
    train_ratings['isbn'] = item_encoder.fit_transform(train_ratings['isbn'])

    # 테스트 데이터는 학습된 인코더로 변환만 수행
    test_ratings['user_id'] = user_encoder.transform(test_ratings['user_id'])
    test_ratings['isbn'] = item_encoder.transform(test_ratings['isbn'])

    num_users = train_ratings['user_id'].nunique()
    num_items = train_ratings['isbn'].nunique()

    # 훈련, 검증, 테스트 세트 분할 (80:10:10 비율)
    train_data, test_data = train_test_split(train_ratings, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    train_dataset = RatingsDataset(train_data)
    val_dataset = RatingsDataset(val_data)
    test_dataset = RatingsDataset(test_data)
    predict_test_dataset = RatingsDataset(test_ratings)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_val)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predict_test_loader = DataLoader(predict_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, predict_test_loader, num_users, num_items
