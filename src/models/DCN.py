import torch.nn as nn
import torch

class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        
        self.w = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        
        self.b = nn.ModuleList([nn.Parameter(torch.empty((input_dim,))) for _ in range(num_layers)])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data) # input tensor를 xavier 초기값으로
            if isinstance(m, nn.parameter):
                nn.init.constant_(m.bias.data, 0) # input tensor를 val값으로 채운다.

    def forward(self, x: torch.Tensor):
        x0 = x
        for i in range(self.num_layers):
            x_w = self.w[i](x)
            x = x0 * x_w + self.b[i] + x
        return x