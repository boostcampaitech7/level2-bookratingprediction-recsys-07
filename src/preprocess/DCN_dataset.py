import regex
import os
import pandas as pd
import numpy as np

# def str2list(x: str) -> list:
#     return x[1:-1].split(', ')


# def split_location(x: str) -> list:
#     '''
#     Parameters
#     ----------
#     x : str
#         location 데이터

#     Returns
#     -------
#     res : list
#         location 데이터를 나눈 뒤, 정제한 결과를 반환합니다.
#         순서는 country, state, city, ... 입니다.
#     '''
#     res = x.split(',')
#     res = [i.strip().lower() for i in res]
#     res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]  # remove special characters
#     res = [i if i not in ['n/a', ''] else np.nan for i in res]  # change 'n/a' into NaN
#     res.reverse()  # reverse the list to get country, state, city, ... order

#     for i in range(len(res)-1, 0, -1):
#         if (res[i] in res[:i]) and (not pd.isna(res[i])):  # remove duplicated values if not NaN
#             res.pop(i)

#     return res

# def pre_context_data(file_path):
#     users = pd.read_csv(os.path.join(file_path,'users.csv'))
#     books = pd.read_csv(os.path.join(file_path,'books.csv'))
#     train = pd.read_csv(os.path.join(file_path,'train_ratings.csv'))
#     test = pd.read_csv(os.path.join(file_path, 'test_ratings.csv'))
#     sub = pd.read_csv(os.path.join(file_path, 'test_ratings.csv'))

#     books['category'] = books['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else np.nan)
#     books['language'] = books['language'].fillna(books['language'].mode()[0])
#     books['publication_range'] = books['year_of_publication'].apply(lambda x: x // 10 * 10)

#     users['age'] = users['age'].fillna(users['age'].mode()[0])
#     users['age_range'] = users['age'].apply(lambda x: x // 10 * 10)

#     users['location_list'] = users['location'].apply(lambda x: split_location(x)) 
#     users['location_country'] = users['location_list'].apply(lambda x: x[0])
#     users['location_state'] = users['location_list'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
#     users['location_city'] = users['location_list'].apply(lambda x: x[2] if len(x) > 2 else np.nan)

#     for idx, row in users.iterrows():
#         if (not pd.isna(row['location_state'])) and pd.isna(row['location_country']):
#             fill_country = users[users['location_state'] == row['location_state']]['location_country'].mode()
#             fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
#             users.loc[idx, 'location_country'] = fill_country
#         elif (not pd.isna(row['location_city'])) and pd.isna(row['location_state']):
#             if not pd.isna(row['location_country']):
#                 fill_state = users[(users['location_country'] == row['location_country']) 
#                                     & (users['location_city'] == row['location_city'])]['location_state'].mode()
#                 fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
#                 users.loc[idx, 'location_state'] = fill_state
#             else:
#                 fill_state = users[users['location_city'] == row['location_city']]['location_state'].mode()
#                 fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
#                 fill_country = users[users['location_city'] == row['location_city']]['location_country'].mode()
#                 fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
#                 users.loc[idx, 'location_country'] = fill_country
#                 users.loc[idx, 'location_state'] = fill_state

#     users = users.drop(['location'], axis=1)

#     user_features = ['user_id', 'age_range', 'location_country', 'location_state', 'location_city']
#     book_features = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'category', 'publication_range']
#     sparse_cols = user_features + book_features

#     train_df = train.merge(users, on='user_id', how='left')\
#                     .merge(books, on='isbn', how='left')[sparse_cols + ['rating']]
#     test_df = test.merge(users, on='user_id', how='left')\
#                   .merge(books, on='isbn', how='left')[sparse_cols]
#     all_df = pd.concat([train_df, test_df], axis=0)

#     label2idx, idx2label = {}, {}
#     for col in sparse_cols:
#         all_df[col] = all_df[col].fillna('unknown')
#         unique_labels = all_df[col].astype("category").cat.categories
#         label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
#         idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
#         train_df[col] = pd.Categorical(train_df[col], categories=unique_labels).codes
#         train_df[col] = train_df[col].replace(-1, label2idx[col].get('unknown', 0))
#         test_df[col] = pd.Categorical(test_df[col], categories=unique_labels).codes
#         test_df[col] = test_df[col].replace(-1, label2idx[col].get('unknown', 0))

    
#     field_dims = [len(label2idx[col]) for col in train_df.columns if col != 'rating']

#     data = {
#             'train': train_df,
#             'test': test_df,
#             'field_names': sparse_cols,
#             'field_dims': field_dims,
#             'label2idx': label2idx,
#             'idx2label': idx2label,
#             'sub': sub,
#             }

#     return data

def pre_context_data(file_path):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    
    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다
    """

    ######################## DATA LOAD
    # users = pd.read_csv(args.dataset.data_path + 'users.csv')
    # books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train_df = pd.read_csv(os.path.join(file_path,'train_ratings.csv'))
    test_df = pd.read_csv(os.path.join(file_path, 'test_ratings.csv'))
    sub = pd.read_csv(os.path.join(file_path, 'test_ratings.csv'))

    all_df = pd.concat([train_df, test_df], axis=0)
    
    sparse_cols = ['user_id', 'isbn']

    # 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = pd.Categorical(train_df[col], categories=unique_labels).codes
        test_df[col] = pd.Categorical(test_df[col], categories=unique_labels).codes
    field_dims = [len(label2idx[col]) for col in sparse_cols]

    data = {
            'train':train_df,
            'test':test_df.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            'sub':sub,
            }


    return data