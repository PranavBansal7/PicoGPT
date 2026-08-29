import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(x.shape[0],x.shape[1],self.num_heads,self.head_dim)
        k = k.reshape(x.shape[0],x.shape[1],self.num_kv_heads,self.head_dim)
        v = v.reshape(x.shape[0],x.shape[1],self.num_kv_heads,self.head_dim)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        k = torch.repeat_interleave(k,self.num_heads//self.num_kv_heads,dim=1)
        v = torch.repeat_interleave(v,self.num_heads//self.num_kv_heads,dim=1)
        score = (q @ k.transpose(-2,-1))/(self.head_dim**0.5)
        mask = torch.tril(torch.ones(x.shape[1],x.shape[1]))
        score = score.masked_fill(mask==0,float('-inf'))
        score = torch.softmax(score,dim=-1)
        att = score @ v
        att = att.transpose(1,2)
        att = att.reshape(x.shape[0],x.shape[1],self.num_heads*self.head_dim)
        att = self.output_proj(att)
        return torch.round(att,decimals=4)


