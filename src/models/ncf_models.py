import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_layers=[64, 32, 16], dropout_rate=0.3):
        super(NCF, self).__init__()

        # 사용자와 아이템 임베딩 층
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP 층 정의
        layers = []
        input_size = embedding_dim * 2  # 사용자와 아이템 임베딩 결합 크기
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # 드롭아웃 추가
            input_size = layer_size
        self.mlp = nn.Sequential(*layers)

        # 최종 출력 층
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

    def forward(self, user_id, item_id):
        # 사용자와 아이템 임베딩
        user_embed = self.user_embedding(user_id)   # (batch_size, embedding_dim)
        item_embed = self.item_embedding(item_id)   # (batch_size, embedding_dim)

        # 사용자와 아이템 임베딩 결합
        combined = torch.cat([user_embed, item_embed], dim=-1)

        # MLP를 통해 비선형 관계 학습
        mlp_output = self.mlp(combined)

        # 최종 예측값
        output = self.output_layer(mlp_output)
        return output.squeeze()


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_layers=[64, 32, 16], dropout_rate=0.3):
        super(NeuMF, self).__init__()

        # GMF용 임베딩 층
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP용 임베딩 층 (GMF와 독립적)
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP 층 정의
        layers = []
        input_size = embedding_dim * 2  # 사용자와 아이템 임베딩 결합 크기
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # 드롭아웃 추가
            input_size = layer_size
        self.mlp = nn.Sequential(*layers)

        # GMF 가중치 레이어
        self.gmf_fc = nn.Linear(embedding_dim, embedding_dim)

        # 최종 결합된 출력을 위한 출력층 (GMF와 MLP 결합)
        self.output_layer = nn.Linear(hidden_layers[-1] + embedding_dim, 1)

    def forward(self, user_id, item_id):
        # GMF: 사용자와 아이템 임베딩을 요소별 곱 후 가중치 레이어 적용
        gmf_user_embed = self.gmf_user_embedding(user_id)
        gmf_item_embed = self.gmf_item_embedding(item_id)
        gmf_output = gmf_user_embed * gmf_item_embed  # 요소별 곱
        gmf_output = self.gmf_fc(gmf_output)  # 가중치 레이어 추가

        # MLP: 사용자와 아이템 임베딩을 결합 후 MLP 통과
        mlp_user_embed = self.mlp_user_embedding(user_id)
        mlp_item_embed = self.mlp_item_embedding(item_id)
        mlp_combined = torch.cat([mlp_user_embed, mlp_item_embed], dim=-1)
        mlp_output = self.mlp(mlp_combined)

        # GMF와 MLP 출력을 결합
        combined_output = torch.cat([gmf_output, mlp_output], dim=-1)

        output = self.output_layer(combined_output)
        return output.squeeze()
