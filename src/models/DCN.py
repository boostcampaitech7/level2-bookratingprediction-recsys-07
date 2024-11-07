import torch.nn as nn
import torch
from numpy import cumsum


class MLP_Base(nn.Module):
    def __init__(self, input_dim, embed_dims, batchnorm=True, dropout=0.2, output_layer=False):
        super().__init__()
        self.mlp = nn.Sequential()
        for idx, embed_dim in enumerate(embed_dims):
            self.mlp.add_module(f'linear{idx}', nn.Linear(input_dim, embed_dim))
            if batchnorm:
                self.mlp.add_module(f'batchnorm{idx}', nn.BatchNorm1d(embed_dim))
            self.mlp.add_module(f'relu{idx}', nn.ReLU())
            if dropout > 0:
                self.mlp.add_module(f'dropout{idx}', nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            self.mlp.add_module('output', nn.Linear(input_dim, 1))
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.mlp(x)
        
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims:list, embed_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = [0, *cumsum(field_dims)[:-1]]

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
        
    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        
        return self.embedding(x)

class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        
        self.w = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        
        self.b = nn.ParameterList([nn.Parameter(torch.empty((input_dim,))) for _ in range(num_layers)])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data) #tensor를 xavier 초기값으로
            if isinstance(m, nn.Parameter):
                nn.init.constant_(m.bias.data, 0) #tensor를 val값으로 채운다.

    def forward(self, x: torch.Tensor):
        x0 = x
        for i in range(self.num_layers):
            x_w = self.w[i](x)
            x = x0 * x_w + self.b[i] + x
        return x

class DeepCrossNetwork(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.cross_network = CrossNetwork(self.embed_output_dim, args.cross_layer_num)
        self.mlp = MLP_Base(self.embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout)
        self.output_layer = nn.Linear(args.mlp_dims[-1], 1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x: torch.Tensor):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        cross_output = self.cross_network(embed_x)
        mlp_output = self.mlp(cross_output)
        predictions = self.output_layer(mlp_output)
        return predictions.squeeze(1)