from dataclasses import dataclass


@dataclass
class Config:
    vocab_size : int = 50304
    n_embd : int = 768
    n_head : int  = 12
    seq_length : int = 1024
    dropout : float = 0.0
    n_layer : int = 12
    bias : bool = True