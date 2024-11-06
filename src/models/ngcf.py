"""
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)

https://github.com/huangtinglin/NGCF-PyTorch/blob/master/NGCF/NGCF.py
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class GNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.W1 = nn.Linear(in_feats, out_feats)
        self.W2 = nn.Linear(in_feats, out_feats)

    def forward(self, L, SelfLoop, feats):
        # (L+I)EW_1
        sf_L = L + SelfLoop
        L = L.cuda()
        sf_L = sf_L.cuda()
        sf_E = torch.sparse.mm(sf_L, feats)
        left_part = self.W1(sf_E)  # left part

        # EL odot EW_2, odot indicates element-wise product
        LE = torch.sparse.mm(L, feats)
        E = torch.mul(LE, feats)
        right_part = self.W2(E)

        return left_part + right_part


class NGCF(nn.Module):
    def __init__(self, args, matrix):
        super(NGCF, self).__init__()
        self.num_users = args.num_users
        self.num_items = args.num_items
        self.latent_dim = args.latent_dim
        self.num_layers = args.num_layers
        self.device = args.device

        self.user_emb = nn.Embedding(self.num_users, self.latent_dim)
        self.item_emb = nn.Embedding(self.num_items, self.latent_dim)

        self.L = self.LaplacianMatrix(matrix)
        self.SL = self.SelfLoop(self.num_users + self.num_items)

        self.leakyrelu = nn.LeakyReLU()
        self.GNNLayers = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.GNNLayers.append(GNNLayer(self.latent_dim, self.latent_dim))

        self.fc_layer = nn.Sequential(
            nn.Linear(self.latent_dim * self.num_layers * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def SelfLoop(self, num):
        i = torch.LongTensor([[k for k in range(0, num)], [j for j in range(0, num)]])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val)

    def LaplacianMatrix(self, ratings):
        iids = ratings["item_id"] + self.num_users
        matrix = sp.coo_matrix((ratings["rating"], (ratings["user_id"], ratings["item_id"])))

        upper_matrix = sp.coo_matrix((ratings["rating"], (ratings["user_id"], iids)))
        lower_matrix = matrix.transpose()
        lower_matrix.resize((self.num_items, self.num_users + self.num_items))

        A = sp.vstack([upper_matrix, lower_matrix])
        row_sum = (A > 0).sum(axis=1)
        # row_sum = np.array(row_sum).flatten()
        diag = list(np.array(row_sum.flatten())[0])
        # D = np.power(diag, -0.5)
        D = np.power(np.array(diag) + 1e-10, -0.5)
        D = sp.diags(D)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        idx = np.stack([row, col])
        idx = torch.LongTensor(idx)
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(idx, data)
        return SparseL

    def FeatureMatrix(self):
        uids = torch.LongTensor([i for i in range(self.num_users)]).to(self.device)
        iids = torch.LongTensor([i for i in range(self.num_items)]).to(self.device)
        user_emb = self.user_emb(uids)
        item_emb = self.item_emb(iids)
        features = torch.cat([user_emb, item_emb], dim=0)
        return features

    def forward(self, uids, iids):
        iids = self.num_users + iids

        features = self.FeatureMatrix()
        final_emb = features.clone()

        for gnn in self.GNNLayers:
            features = gnn(self.L, self.SL, features)
            features = self.leakyrelu(features)
            final_emb = torch.concat([final_emb, features], dim=-1)

        user_emb = final_emb[uids]
        item_emb = final_emb[iids]

        inputs = torch.concat([user_emb, item_emb], dim=-1)
        outs = self.fc_layer(inputs)
        return outs.flatten()
