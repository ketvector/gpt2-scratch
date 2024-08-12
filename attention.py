import torch
from torch import nn

from config import Config
from typing_extensions import Self

"""
convert (B,T,C) tensor to (B, H, T, HS) 
"""
def get_multi_head_tensor(x: torch.Tensor, config: Config, B: int, T: int, C: int):
    single_head_size = C // config.n_head
    return x.view(B, T, config.n_head, single_head_size).transpose(1,2)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super(CausalSelfAttention,self).__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)


        sl = config.seq_length
        ones = torch.ones(sl, sl).view(1,1,sl,sl)
        self.register_buffer("bias", torch.tril(ones))


        
    def forward(self: Self, x: torch.Tensor):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=2)

        
        q = get_multi_head_tensor(q, self.config, B, T, C)
        k = get_multi_head_tensor(k, self.config, B, T, C)
        v = get_multi_head_tensor(v, self.config, B, T, C)

        mm = q @ k.transpose(-2,-1)
        s = mm / torch.sqrt(torch.tensor(k.size(-1)))
        mskd =  s.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        smx = nn.functional.softmax(mskd, dim=-1)
        attn = smx @ v
        attn = self.attn_dropout(attn)

        attn = attn.transpose(1,2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(attn))

        return y





