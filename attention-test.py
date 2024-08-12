import torch

from attention import CausalSelfAttention
from config import Config


B = 32
T = 1024
C = 768


c_attn = CausalSelfAttention(Config())
sample = torch.ones([B,T,C])
out = c_attn.forward(sample).size()


print(out)