from einops import einsum, rearrange, reduce, repeat
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math
import numpy as np
import os
import typing
from collections.abc import Callable, Iterable
from typing import Optional
class Linear(nn.Module): 
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()  
        sigma = 2/ (in_features * out_features)
        W = nn.Parameter(torch.randn(size = (out_features, in_features), 
                                device =device, dtype = dtype, requires_grad = True)* sigma )
        self.W = nn.init.trunc_normal_(W, a = -3*sigma, b = 3*sigma)
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        return einsum(x, self.W , "... feature_in, feature_out feature_in-> ... feature_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()  
        emb_mat =  nn.Parameter(torch.randn(size =(num_embeddings, embedding_dim),
                    device =device, dtype = dtype, requires_grad = True ))
        self.emb_mat = nn.init.trunc_normal_(emb_mat, a = -3, b = 3)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        return self.emb_mat[token_ids]

class RMSNorm(nn.Module): 
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None): 
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.randn(size =(d_model, ) ,
                    device =device, dtype = dtype, requires_grad = True ))
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        in_type = x.dtype
        x = x.to(torch.float32)
        squares = einsum(x, x, "... d, ... d -> ... d ")
        rms  = torch.sqrt(reduce(squares, "... d -> ... 1", "mean") +self.eps)
        rmsnorm =  einsum(x/rms, self.g, "... d, d -> ... d") 
        return rmsnorm.to(in_type)
     
    
class SwiGLU(nn.Module): 
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)
        self.W3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) ->torch.Tensor: 
        w1x = self.W1(x) 
        SiLU = w1x * torch.sigmoid(w1x) 
        w3x = self.W3(x)
        SiLUW3 = einsum(SiLU , w3x , "..., ... -> ...")
        result = self.W2(SiLUW3)
        return result

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None): 
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k 
        self.max_seq_len = max_seq_len 
        i = torch.arange(start =0, end = max_seq_len, device = device, dtype=torch.float32)
        i_k = torch.arange(start =1, end =d_k//2 +1 , device = device, dtype=torch.float32)

        freq =  1/ theta ** ((2* i_k -2)/ d_k)

        theta_tensor = einsum(i, freq, " max_seq, d_k -> max_seq d_k ")
        
        theta_tensor = repeat(theta_tensor, "max_seq d_k -> max_seq (r d_k)", r=2)
        
        self.register_buffer("cos", theta_tensor.cos(), persistent=False)
        self.register_buffer("sin", theta_tensor.sin(), persistent=False)

 
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1, x2 = rearrange(x , "... (n d_2) -> n ... d_2", n =2)
        x_rotated = rearrange([-x2 , x1], " n ... d_2 -> ... ( n d_2)")
        out = einsum(cos, x, "... s d, ... s d -> ... s d ") + einsum(sin, x_rotated, "... s d, ... s d -> ... s d ")
        return out
def softmax(x: torch.Tensor, dim: int): 
    x_normal = x - torch.max(x, dim = dim, keepdim =True).values
    e_x = torch.exp(x_normal)
    return e_x / e_x.sum(dim = dim, keepdim =True)

    
def scaled_dot_product_attention(k: torch.Tensor, q: torch.Tensor, v: torch.Tensor, mask : torch.Tensor | None ) -> torch.Tensor:
    
    q_k_dot = einsum(q, k, "... seq_len1 d_k, ... seq_len2 d_k -> ... seq_len1 seq_len2")
    
    if mask is not None: 
        while mask.ndim < q_k_dot.ndim:
            mask = mask.unsqueeze(0) 

        q_k_dot = q_k_dot.masked_fill( mask = mask == 0, value = -1e9)

       
    softmax_q_k = torch.softmax(q_k_dot /math.sqrt(k.shape[-1]), dim =-1)
    attn = einsum(softmax_q_k, v, "... seq_len1 seq_len2, ... seq_len2 d_v -> ... seq_len1 d_v" )
    
    return attn
class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, d_model:int, num_heads: int,
                 rope: nn.Module | None = None,
                token_positions: torch.Tensor | None = None):
        super().__init__()
        """ 
        Q: d_model d_model
        K: d_model d_model
        V:  d_model d_model
        """
        self.rope = rope
        self.token_positions = token_positions
        self.d_model = d_model
        self.num_heads = num_heads
 
        self.WQKV = Linear(d_model, 3*d_model)
        self.WO =  Linear(d_model, d_model)

    def forward(self, x: torch.Tensor): 

        WQKV = self.WQKV(x)
        Q, K, V  = rearrange(WQKV, "... seq_len ( n n_heads d) -> n ... n_heads seq_len d", 
                n =3, n_heads  =  self.num_heads, d = self.d_model //  self.num_heads)
        if self.rope is not None: 
            rope = self.rope
            if self.token_positions is not None:
                Q= rope(Q, self.token_positions)
                K = rope(K, self.token_positions)
         
            else: 
                token_positions = torch.arange(Q.shape[-2], device =Q.device) 
                Q= rope(Q, token_positions)
                K = rope(K, token_positions)
            
        seq_len = V.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device = V.device))
        
        attn = rearrange(scaled_dot_product_attention(q =Q, k =K, v = V, mask = mask), 
                        "... n_heads seq_len d -> ... seq_len (n_heads d)" )
        
        return self.WO(attn)


"""
    
    "transformer" :  
    {"vocab_size": 50257,
    "context_length": 1024,
    "num_layers": 48,
    "d_model": 1600,
    "num_heads": 25,
    "d_ff": 4288,
    "rope_theta": 10000.0,
    "device" : 0
    
    }
    "training": {
    "batch_size": 16,
    "lr_max": 0.0005,
    "lr_min": 0.0,
    "weight_decay": 0.1,
    "warmup_iters": 1000,
    "cosine_cycle_iters": 100000,
    "max_l2_norm": 1.0,
    }
"""
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope, device):
        super().__init__()
        # 1. First sub-block (Attention) components
        self.attn_norm = RMSNorm(d_model, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope)
        
        # 2. Second sub-block (Feed-Forward) components
        self.ffn_norm = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        # First residual block: Pre-LN Attention shortcut
        x = x + self.attn(self.attn_norm(x))
        
        # Second residual block: Pre-LN SwiGLU shortcut
        x = x + self.ffn(self.ffn_norm(x))
        return x
class Transformer(nn.Module): 
    def __init__(self, vocab_size, context_length, num_layers, 
    d_model, num_heads, d_ff,  rope_theta, device): 
        super().__init__()
        self.emb = Embedding(vocab_size, d_model)
        rope = RotaryPositionalEmbedding(theta = rope_theta,
         d_k = d_model // num_heads
         , max_seq_len = context_length)


        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, d_ff, rope, device) 
            for _ in range(num_layers)
        ])
        self.final_rms = RMSNorm(d_model, device = device)
        self.LM_head = Linear(d_model, vocab_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x_emb = self.emb(x)
        x_tb = self.transformer_blocks(x_emb)
        x_norm  =self.final_rms(x_tb)
        out = self.LM_head(x_norm)
        return out

rms = RotaryPositionalEmbedding(d_k = 16, theta = 2, max_seq_len = 12 )
logits = torch.randn(3, 4, requires_grad=True)

targets = torch.tensor([1, 0, 3])


 