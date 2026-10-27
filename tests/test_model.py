import torch
import pytest
from transformer_foundations.transformer import Transformer, RMSNorm, SwiGLU

def test_transformer_forward():
    vocab_size = 1000
    context_length = 64
    d_model = 128
    num_heads = 4
    num_layers = 2
    d_ff = 512
    rope_theta = 10000.0

    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device="cpu"
    )

    x = torch.randint(0, vocab_size, (2, 32))
    logits = model(x)

    assert logits.shape == (2, 32, vocab_size)

def test_rmsnorm():
    norm = RMSNorm(d_model=128)
    x = torch.randn(2, 32, 128)
    out = norm(x)
    assert out.shape == (2, 32, 128)

def test_swiglu():
    ffn = SwiGLU(d_model=128, d_ff=512)
    x = torch.randn(2, 32, 128)
    out = ffn(x)
    assert out.shape == (2, 32, 128)
