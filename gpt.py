import torch
from torch import nn

from transformers import GPT2LMHeadModel

from config import Config
from block import Block 

class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.seq_length, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        self.transformer.wte.weight = self.lm_head.weight

        print(f"num params : {self.get_num_params()/1e6}")
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, idx, targets=None):
        B, T  = idx.size()

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        we = self.transformer.wte(idx)
        pe = self.transformer.wpe(pos)

        x = we + pe
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)


        if targets != None:
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return x, loss

        else:
            logits = self.lm_head(x[: , [-1], :])
            return logits, None
        
    @classmethod
    def from_pretrained(cls):
        # n_layer, n_head and n_embd are determined from model_type
        config_args = dict(n_layer=12, n_head=12, n_embd=768)
        print("forcing vocab_size=50257, seq_length=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['seq_length'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        config = Config(**config_args)
        model = GPT(config)
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

        
