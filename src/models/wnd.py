import torch
import torch.nn as nn


class WideAndDeep(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(WideAndDeep, self).__init__()
        # Wide part: Linear layer for wide features
        self.wide = nn.EmbeddingBag(sum(field_dims), 1, mode='sum')
        
        # Deep part: Embedding and MLP layers
        self.embeddings = nn.ModuleList([nn.Embedding(field_dim, embed_dim) for field_dim in field_dims])
        self.deep = nn.Sequential(
            nn.Linear(len(field_dims) * embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Wide part
        wide_part = self.wide(x)

        # Deep part
        embeds = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]
        deep_part = torch.cat(embeds, dim=1)
        deep_part = self.deep(deep_part)

        # Combine wide and deep part
        output = wide_part + deep_part
        return output.squeeze(1)