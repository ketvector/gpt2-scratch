from gpt import GPT
from config import Config

import torch
import torch.nn.functional as F

import tiktoken

model = GPT.from_pretrained()
#model = GPT(Config())

idx = []
encoder = tiktoken.encoding_for_model("gpt-2")
for i in range(5):
    text = "let's go out and play with"
    encoded = encoder.encode(text)
    #print(encoded)
    idx.append(encoded)


idx = torch.tensor(idx)
max_len = 30
i = len(encoded)

while i < max_len:
    output, _ = model(idx)
    #print(output.size())

    sftmax = F.softmax(output, dim=-1)
    #print(sftmax.size())

    #mx = torch.argmax(sftmax, dim=-1)
    #print(mx)
    mx = torch.multinomial(sftmax.squeeze(1), num_samples=1)

    #token = encoder.decode([mx[0].item()])
    
    idx = torch.cat([idx, mx], dim=1)

    i = i + 1

print(idx)
for i,x in enumerate(idx):
    print(f"{i}. {encoder.decode(list(x))} \n")



