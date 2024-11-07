import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 상위 폴더에 있는 파일 import
try:
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
except Exception as e:
    print(e)
    sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

from models.ngcf import NGCF
from preprocess.ngcf_dataset import GraphDataset


def main(dir_path, args):
    # 데이터 파일 경로 설정
    train_path = f"{dir_path}/train_ratings.csv"
    test_path = f"{dir_path}/test_ratings.csv"

    # 훈련 데이터 로드 및 전처리
    df_train = pd.read_csv(train_path)
    df_train["rating"] = df_train["rating"]
    df_train = df_train.rename(columns={"isbn": "item_id"})
    df_train = df_train.astype({"user_id": "category", "item_id": "category"})
    df_train["_type"] = "train"

    # 테스트 데이터 로드 및 전처리
    df_test = pd.read_csv(test_path)
    df_test["rating"] = df_test["rating"]
    df_test = df_test.rename(columns={"isbn": "item_id"})
    df_test = df_test.astype({"user_id": "category", "item_id": "category"})
    df_test["_type"] = "test"

    # 훈련 및 테스트 데이터 결합
    df = pd.concat([df_train, df_test])
    df = df.astype({"user_id": "category", "item_id": "category"})

    # 훈련 데이터를 훈련셋과 검증셋으로 분할
    df_train, df_val = train_test_split(df_train, train_size=0.8, random_state=args.seed)

    # 데이터 타입 변환
    df_train = df_train.astype({"user_id": "category", "item_id": "category"})
    df_val = df_val.astype({"user_id": "category", "item_id": "category"})
    df_test = df_test.astype({"user_id": "category", "item_id": "category"})

    # 사용자 및 아이템 카테고리 설정
    u_cat = df.user_id.cat.categories
    i_cat = df.item_id.cat.categories

    # 카테고리 설정
    df_train.user_id = df_train.user_id.cat.set_categories(u_cat)
    df_train.item_id = df_train.item_id.cat.set_categories(i_cat)

    df_val.user_id = df_val.user_id.cat.set_categories(u_cat)
    df_val.item_id = df_val.item_id.cat.set_categories(i_cat)

    df_test.user_id = df_test.user_id.cat.set_categories(u_cat)
    df_test.item_id = df_test.item_id.cat.set_categories(i_cat)

    # 카테고리를 코드로 변환
    df.user_id = df.user_id.cat.codes
    df.item_id = df.item_id.cat.codes

    df_train.user_id = df_train.user_id.cat.codes
    df_train.item_id = df_train.item_id.cat.codes

    df_val.user_id = df_val.user_id.cat.codes
    df_val.item_id = df_val.item_id.cat.codes

    df_test.user_id = df_test.user_id.cat.codes
    df_test.item_id = df_test.item_id.cat.codes

    # 데이터 타입을 정수로 변환
    df = df.astype({"user_id": int, "item_id": int})
    df_train = df_train.astype({"user_id": int, "item_id": int})
    df_val = df_val.astype({"user_id": int, "item_id": int})
    df_test = df_test.astype({"user_id": int, "item_id": int})

    # 사용자 및 아이템 수 설정
    args.num_users = df.user_id.max() + 1
    args.num_items = df.item_id.max() + 1

    # 데이터셋 및 데이터로더 생성
    train_dataset = GraphDataset(df_train)
    val_dataset = GraphDataset(df_val)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 모델, 옵티마이저, 손실 함수 초기화
    model = NGCF(args, df).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    print("Start Train...")
    best_loss = float("inf")
    train_losses, valid_losses = [], []

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss = 0.0
        model.train()
        for batch in tqdm(train_dataloader, desc="training..."):
            batch = tuple(b.to(args.device) for b in batch)
            inputs = {"uids": batch[0], "iids": batch[1]}
            gt_ratings = batch[2].float()

            pred_ratings = model(**inputs)

            loss = criterion(pred_ratings, gt_ratings)

            loss = torch.sqrt(loss)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # Val
        gt_ratings_list = []
        pred_ratings_list = []
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="evaluating..."):
                batch = tuple(b.to(args.device) for b in batch)
                inputs = {"uids": batch[0], "iids": batch[1]}
                gt_ratings = batch[2].float()

                pred_ratings = model(**inputs)

                loss = criterion(pred_ratings, gt_ratings)
                loss = torch.sqrt(loss)
                val_loss += loss.item()

                # RMSE 계산을 위해 실제 값과 예측 값 저장
                gt_ratings_list.extend(gt_ratings.cpu().numpy())
                pred_ratings_list.extend(pred_ratings.cpu().numpy())

        val_loss /= len(val_dataloader)
        valid_losses.append(val_loss)

        # RMSE 계산
        val_rmse = np.sqrt(np.mean((np.array(gt_ratings_list) - np.array(pred_ratings_list)) ** 2))

        print(f"[{epoch}/{args.num_epochs}]: ", end=" ")
        print(f"Train loss: {train_loss:.4f}\t Val loss: {val_loss:4f}\t Val RMSE: {val_rmse:4f}")

        # Save best model
        if best_loss > val_loss:
            best_loss = val_loss
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model, os.path.join(args.save_path, f"{model._get_name()}_model.pt"))

    print("Test...")
    # test dataset
    test_dataset = GraphDataset(df_test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    sample_path = f"{dir_path}/sample_submission.csv"
    df_sample = pd.read_csv(sample_path)

    # Load saved model
    model = NGCF(args, df).to(args.device)
    model = torch.load(os.path.join(args.save_path, f"{model._get_name()}_model.pt"))

    # 사용자 및 아이템 ID 매핑
    user_id_map = dict(enumerate(u_cat))
    item_id_map = dict(enumerate(i_cat))
    df_test["uid"] = df_test["user_id"].map(user_id_map)
    df_test["iid"] = df_test["item_id"].map(item_id_map)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="test..."):
            batch = tuple(b.to(args.device) for b in batch)
            inputs = {"uids": batch[0], "iids": batch[1]}

            pred_ratings = model(**inputs)

            # 배치 단위로 예측 결과 업데이트
            for uid, iid, pred_rating in zip(
                batch[0].cpu().numpy(), batch[1].cpu().numpy(), pred_ratings.cpu().numpy()
            ):
                user_id = user_id_map[uid]
                isbn = item_id_map[iid]
                df_sample.loc[(df_sample["user_id"] == user_id) & (df_sample["isbn"] == isbn), "rating"] = int(
                    pred_rating
                )

    # Save results
    df_sample["rating"] = df_sample["rating"].clip(
        lower=0, upper=10
    )  # 예측 결과 후처리: rating을 0~10 사이의 값으로 제한함
    df_sample.to_csv("./output.csv", index=False)

    print("END")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")

    # add basic arguments
    parser.add_argument("--seed", type=int, help="랜덤 고정 시드")
    parser.add_argument("--num_layers", type=int, help="레이어 수")
    parser.add_argument("--batch_size", type=int, help="배치 크기")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], help="사용할 디바이스 종류")
    parser.add_argument("--save_path", type=str, default="./", help="모델 저장 경로")
    parser.add_argument("--num_users", type=int, help="사용자 수")
    parser.add_argument("--num_items", type=int, help="아이템 수")
    parser.add_argument("--latent_dim", type=int, help="잠재 차원")
    parser.add_argument("--num_epochs", type=int, help="에폭 수")
    parser.add_argument("--lr", type=float, help="학습률")

    # 인자 파싱 및 설정
    args = parser.parse_args(args=[])
    args.seed = 42
    args.num_layers = 3
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.save_path = "weights"
    args.latent_dim = 64
    args.batch_size = 256
    args.num_epochs = 200
    args.lr = 0.00001

    print(f"args: {args}")

    dir_path = "/data/ephemeral/home/yang/level2-bookratingprediction-recsys-07/data"

    main(dir_path, args)
