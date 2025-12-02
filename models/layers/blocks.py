import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MLP(nn.Module):
    """
    [修改] 将 BatchNorm1d 替换为 LayerNorm，防止 Batch=1 或 Edge=1 时 NaN
    """
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList() # 改名 batch_norms -> norms

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                # 使用 LayerNorm 替代 BatchNorm1d
                self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                # LayerNorm 位置
                h = F.relu(self.norms[i](self.linears[i](h)))
            return self.linears[-1](h)

class EdgeConv(nn.Module):
    """
    [修改] 内部 batchnorm 改为 LayerNorm
    """
    def __init__(self, edge_feats, out_feats, node_feats, num_mlp_layers=2, hidden_mlp_dim=64):
        super().__init__()
        self.proj = MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        
        # 修改为 LayerNorm
        self.norm = nn.LayerNorm(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        src, dst = graph.edges()
        src = src.to(torch.long)
        dst = dst.to(torch.long)
        
        # 增加保护：如果图没有边，直接返回零张量 (防止 MLP 报错)
        if src.numel() == 0:
            return torch.zeros((0, self.norm.normalized_shape[0]), device=nfeat.device, dtype=nfeat.dtype)

        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])
        agg = proj1 + proj2
        h = self.mlp((1 + self.eps) * efeat + agg)
        
        # 使用 LayerNorm
        h = F.leaky_relu(self.norm(h), inplace=True)
        return h

class NonLinearClassifier(nn.Module):
    """
    [修改] BatchNorm1d -> LayerNorm
    """
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.ln1 = nn.LayerNorm(512) # BN -> LN
        self.dp1 = nn.Dropout(p=dropout)
        
        self.linear2 = nn.Linear(512, 512, bias=False)
        self.ln2 = nn.LayerNorm(512) # BN -> LN
        self.dp2 = nn.Dropout(p=dropout)
        
        self.linear3 = nn.Linear(512, 256, bias=False)
        self.ln3 = nn.LayerNorm(256) # BN -> LN
        self.dp3 = nn.Dropout(p=dropout)
        
        self.linear4 = nn.Linear(256, num_classes)
        
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.ln1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.ln2(self.linear2(x)))
        x = self.dp2(x)
        x = F.relu(self.ln3(self.linear3(x)))
        x = self.dp3(x)
        x = self.linear4(x)
        # 确保没有 Softmax
        return x

def init_params_global(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()