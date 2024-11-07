import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm.auto import tqdm

###############################################################################
# 주어진 결과와 정확히 비교하기 위한 random seed 고정
###############################################################################
SEED = 0  # 바꾸지 마시오!
random.seed(SEED)
np.random.seed(SEED)

def reset_and_set_seed(seed=SEED):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

data_path = '../data/raw/'

# 데이터 로드 및 전처리
train_ratings_df = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
test_ratings_df = pd.read_csv(os.path.join(data_path, 'test_ratings.csv'))
books_df = pd.read_csv(os.path.join(data_path, 'books.csv'))
users_df = pd.read_csv(os.path.join(data_path, 'users.csv'))
sample_submission_df = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))

# change N/A values in language to 'en'
books_df['language'] = books_df['language'].fillna('en')

def year_map(x: int) -> int:
    x = int(x)
    return (x // 10) * 10

books_df['publication_decade'] = books_df['year_of_publication'].apply(lambda x: year_map(x))    

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 10
    elif x >= 20 and x < 30:
        return 20
    elif x >= 30 and x < 40:
        return 30
    elif x >= 40 and x < 50:
        return 40
    elif x >= 50 and x < 60:
        return 50
    else:
        return 60

users_df['age_range'] = users_df['age'].apply(lambda x: age_map(x) if not pd.isna(x) else 0)

# location 정보를 리스트로 변환 및 중복 제거
import regex
def split_location(x: str) -> list:
    res = x.split(',')
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]  # remove special characters
    res = [i if i not in ['n/a', ''] else np.nan for i in res]  # change 'n/a' into <NA>
    res.reverse()

    # remove duplicates inside list
    for i in range(len(res)-1, 0, -1):
        if res[i] in res[:i]:
            res.pop(i)

    return res

users_df['location_list'] = users_df['location'].apply(lambda x: split_location(x))

# 주어진 지역 리스트 정보 중 상위 두 개만 사용하도록 함
users_df['location_country'] = users_df['location_list'].apply(lambda x: x[0])
users_df['location_state'] = users_df['location_list'].apply(lambda x: x[1] if len(x) >= 2 else np.nan)
users_df['location_city'] = users_df['location_list'].apply(lambda x: x[2] if len(x) >= 3 else np.nan)

# 만일 지역 정보의 뒷부분(주 또는 도시)은 주어졌으나 지역 정보 앞부분(국가 또는 주)이 없는 경우, 최빈값으로 대체
for idx, row in users_df.iterrows():
    if (not pd.isna(row['location_state'])) and pd.isna(row['location_country']):
        fill_country = users_df[users_df['location_state'] == row['location_state']]['location_country'].mode()
        fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
        users_df.loc[idx, 'location_country'] = fill_country
    elif (not pd.isna(row['location_city'])) and pd.isna(row['location_state']):
        if not pd.isna(row['location_country']):
            fill_state = users_df[(users_df['location_country'] == row['location_country'])
                                & (users_df['location_city'] == row['location_city'])]['location_state'].mode()
            fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
            users_df.loc[idx, 'location_state'] = fill_state
        else:
            fill_state = users_df[users_df['location_city'] == row['location_city']]['location_state'].mode()
            fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
            fill_country = users_df[users_df['location_city'] == row['location_city']]['location_country'].mode()
            fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
            users_df.loc[idx, 'location_country'] = fill_country
            users_df.loc[idx, 'location_state'] = fill_state

user_features = ['user_id', 'age_range', 'location_country', 'location_state', 'location_city']
item_features = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'publication_decade']

df = pd.merge(train_ratings_df, users_df[user_features], on='user_id', how='left')
df = pd.merge(df, books_df[item_features], on='isbn', how='left')

# label encoding
label_dict = dict()
df_label = df.copy()
for col in df_label.columns:
    if col == 'rating':
        continue
    df_label[col] = df_label[col].astype("category")  # 정수값이라 카테고리형 변수로 인식되지 않는 경우가 있으므로 모두 변환
    label_dict[col] = {value: idx for idx, value in enumerate(df_label[col].cat.categories)}
    df_label[col] = df_label[col].cat.codes

    # label 개수 출력
    print(f'{col} : {len(label_dict[col])}')

    # label_dict 일부 출력
    tmp = {k: v for k, v in list(label_dict[col].items())[:10]}
    print(f'\t{tmp}')

train_X, test_X, train_y, test_y = train_test_split(
    df_label.loc[:, df_label.columns != 'rating'], df_label['rating'], test_size=0.2, random_state=SEED)
print('***** FM & FFM 데이터 *****')
print('학습 데이터 크기:', train_X.shape, train_y.shape)
print('테스트 데이터 크기:', test_X.shape, test_y.shape)

train_dataset = TensorDataset(torch.LongTensor(np.array(train_X)), torch.Tensor(np.array(train_y)))
test_dataset = TensorDataset(torch.LongTensor(np.array(test_X)), torch.Tensor(np.array(test_y)))

class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims:list, embed_dim:int):
        """
        Embed the features (sparse, high-dimensional input, label-encoded) into a dense vector of a fixed size.

        Parameters
        ----------
        field_dims : A list of feature dimensions in each field
        embed_dim : Embedding dimension  (Same with `factor_dim`)

        Examples
        --------
        >>> field_dims = [3, 3, 3]
        >>> embed_dim = 2
        >>> x = torch.Tensor([[1, 0, 0], [2, 1, 0], [0, 0, 1]])
        >>> embedding = FeaturesEmbedding(field_dims, embed_dim)
        >>> embedding(x)

        """
        super().__init__()

        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.concatenate([[0], np.cumsum(field_dims)[:-1]])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x:torch.Tensor):
        """
        Parameters
        ----------
        x : Embeddings of features. Long tensor of size ``(batch_size, num_fields)``

        Returns
        -------
        Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return self.embedding(x)

class FMLayer(nn.Module):
    def __init__(self, field_dims:list, factor_dim:int):
        '''
        Factorization Machine Layer using FeaturesEmbedding

        Parameter
        ----------
        field_dims : A list of feature dimensions in each field
        factor_dim : Factorization dimension

        Example
        ----------
        >>> x = torch.Tensor([[1, 0, 0], [2, 1, 0], [0, 0, 1]])
        >>> fm = FMLayer(field_dims=[3, 3, 3], factor_dim=2)
        >>> fm(x)

        '''

        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, factor_dim)

    def square(self, x):
        return torch.pow(x,2)

    def forward(self, x):
        '''
        Parameter
        ----------
        x : Long tensor of size ``(batch_size, num_fields)``

        Return
        ----------
        Float tensor of size ``(batch_size,)``
        '''
        x = self.embedding(x)
        square_of_sum = self.square(torch.sum(x, dim=1))
        sum_of_square = torch.sum(self.square(x), dim=1)

        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)

class FeaturesLinear(nn.Module):
    def __init__(self, field_dims:list, output_dim:int=1, bias=True):
        '''
        Linear Transformation Layer

        Parameter
        ----------
        field_dims : A list of feature dimensions in each field
        output_dim : Output dimension
        bias : Bias term (default: True)

        Example
        ----------
        >>> x = torch.Tensor([[1, 0, 0], [2, 1, 0], [0, 0, 1]])
        >>> linear = FeaturesLinear(field_dims=[3, 3, 3], output_dim=1)
        >>> linear(x)
        '''
        super().__init__()

        self.feature_dim = sum(field_dims)
        self.offsets = np.concatenate([[0], np.cumsum(field_dims)[:-1]])
        self.fc = nn.Linear(self.feature_dim, output_dim, bias=bias)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Long tensor of size ``(batch_size, num_fields)``

        Returns
        -------
        Float tensor of size ``(batch_size, output_dim)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = torch.zeros(x.size(0), self.feature_dim, device=x.device).scatter_(1, x, 1)  # should be aware of the device where x should be in

        return self.fc(x)

class FactorizationMachine(nn.Module):
    def __init__(self, field_dims:list, factor_dim:int):
        '''
        Factorization Machine (FM) Model

        Parameter
        ----------
        field_dims : A list of feature dimensions in each field
        factor_dim : Factorization dimension

        Example
        ----------
        >>> x = torch.Tensor([[1, 0, 0], [2, 1, 0], [0, 0, 1]])
        >>> fm_model = FactorizationMachine(field_dims=[3, 3, 3], factor_dim=2)
        >>> fm_model(x)
        '''
        super(FactorizationMachine, self).__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)
        self.feature_dim = sum(field_dims)
        self.factor_dim = factor_dim

        self.linear = FeaturesLinear(self.field_dims, 1, bias=True)
        self.fm = FMLayer(self.field_dims, self.factor_dim)


    def forward(self, x):
        '''
        Parameter
        ----------
        x : Long tensor of size ``(batch_size, num_fields)``

        Return
        ----------
        Float tensor of size ``(batch_size,)``
        '''
        y = self.linear(x).squeeze(1) + self.fm(x)

        return y

def train_loop(dataloader, model, loss_fn, optimizer, device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for (X, y) in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return None


def eval_loop(dataloader, model, loss_fn, task, name='Test', device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, y_all, pred_all = 0, list(), list()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() / num_batches
            y_all.append(y)
            pred_all.append(pred)

    y_all = torch.cat(y_all).cpu()
    pred_all = torch.cat(pred_all).cpu()

    if task == 'reg':
        err = abs(pred_all - y_all).type(torch.float).mean().item()  # MAE
        rmse = torch.sqrt(((pred_all - y_all) ** 2).mean()).item()  # RMSE
        print(f"{name} Error: \n  MAE: {(err):>8f} \n  RMSE: {rmse:>8f} \n  Avg loss: {test_loss:>8f}")
    else:
        err = roc_auc_score(y_all, torch.sigmoid(pred_all)).item()
        print(f"{name} Error: \n  AUC: {err:>8f} \n  Avg loss: {test_loss:>8f}")

    return err, test_loss

def train_and_test(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, task, device='cpu'):
    train_loss, test_err, test_loss = list(), list(), list()

    for t in range(epochs):
        print(f"Epoch {t+1}")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        print("-------------------------------")
        train_result = eval_loop(train_dataloader, model, loss_fn, task, name='Train', device=device)
        test_result = eval_loop(test_dataloader, model, loss_fn, task, name='Test', device=device)
        train_loss.append(train_result[1])
        test_err.append(test_result[0])
        test_loss.append(test_result[1])
        print("-------------------------------\n")
    print("Done!")

    return train_loss, test_err, test_loss

######### Hyperparameter #########
# 다양한 하이퍼파라미터 조합을 실험해보세요!
# 주의) 주어진 하이퍼파라미터에 대해서 1 epoch 당 약 5초가 소요됩니다. (FMLayer_1를 사용할 경우 약 12초)

batch_size = 1024
data_shuffle = True
task = 'reg'
factorization_dim = 8
epochs = 10
learning_rate = 0.001
weight_decay = 1e-4
gpu_idx = 0

##################################

reset_and_set_seed()
device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")
print(device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=data_shuffle)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=data_shuffle)

field_dims = list(len(label_dict[k]) for k in label_dict)
model = FactorizationMachine(field_dims, factorization_dim).to(device)

loss_fn = nn.MSELoss().to(device) if (task == 'reg') else nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

train_loss, test_err, test_loss = train_and_test(train_dataloader, test_dataloader,
                                                 model, loss_fn, optimizer, epochs, task, device)