# Transformer Foundations: First-Principles Large Language Model Architecture & Parallel Tokenization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Large Language Model (LLM) codebase written from first principles in PyTorch. Implements a decoder-only Transformer architecture, a high-throughput $\mathcal{O}(N \log K)$ Min-Heap BPE Tokenizer, and training/generation CLI utilities.

---

## 🛠️ Codebase Structure & Documentation

### 1. Decoder-Only Transformer Architecture (`src/transformer_foundations/transformer.py` / `Transformer.py`)
- **Linear**: Custom Linear layer initialized with truncated normal weights (`nn.init.trunc_normal_`) using `einops.einsum` for tensor contractions.
- **Embedding**: Embedding matrix layer initialized via truncated normal values.
- **RMSNorm**: Root Mean Square Layer Normalization with FP32 precision accumulation and trainable gain vector `self.g`.
- **SwiGLU**: Swish-Gated Linear Unit feed-forward network combining $W_1$, $W_2$, and $W_3$ projections.
- **RotaryPositionalEmbedding (RoPE)**: Precomputes frequency tensors (`cos`, `sin`) and rotates pairs of dimensions (`x1`, `x2`).
- **MultiHeadSelfAttention**: Fused $W_{QKV}$ linear projections, optional RoPE application, lower-triangular causal masking, and output projection $W_O$.
- **TransformerBlock & Transformer**: Full decoder-only LLM stack combining pre-LN residual shortcuts and language modeling head.

### 2. Parallel BPE Tokenizer & Min-Heap Merging (`src/transformer_foundations/tokenizer.py` & `bpe.py`)
- **Parallel Pre-Tokenization (`bpe.py`)**: Binary `mmap` file chunking (`find_chunk_boundaries()`) with multi-core `ProcessPoolExecutor` regex pre-token frequency counting.
- **Max-Heap BPE Training (`bpe.py`)**: Max-Heap pair merge training (`RevTuple`), inverted word index mapping (`pair_to_word`), and `vocab.pkl` / `merges.pkl` serialization.
- **Min-Heap $\mathcal{O}(N \log K)$ Encoder (`tokenizer.py` / `Tokenizer.py`)**: Fast BPE tokenizer using a Min-Heap priority queue (`heapq`) paired with linked-list index arrays (`prev_indices`, `next_indices`) for $\mathcal{O}(N \log K)$ merges.
- **Binary Serialization**: `tokenizer_and_save_to_file()` streaming token IDs into binary `uint16` datasets for memory-mapped training.

### 3. Training Scaffold & Autoregressive Generation (`train_scaffold.py`, `train.py`, `generate.py`)
- **Loss & Optimizer**: `cross_entropy_loss` using log-sum-exp stabilization, custom `AdamW` optimizer with decoupled weight decay and bias correction.
- **Training Loop (`train.py`)**: `np.memmap` dataset batching, cosine learning rate scheduler with warmup, and global L2 gradient clipping.
- **Sampling CLI (`generate.py`)**: Autoregressive text generation with Temperature scaling, Top-K filtering, and Top-P (nucleus) sampling.

---

## 📐 Repository Tree

```
transformer-foundations/
├── pyproject.toml              # Project dependencies managed via uv
├── README.md                   # Technical documentation
└── src/transformer_foundations/# Core Python Package
    ├── __init__.py
    ├── Transformer.py          # Transformer LM implementation
    ├── Tokenizer.py            # Min-Heap BPE Tokenizer
    ├── bpe.py                  # Parallel BPE training engine
    ├── train_scaffold.py       # AdamW & cosine schedule scaffold
    ├── train.py                # Training loop CLI
    └── generate.py             # Text generation CLI
```

---

## ⚡ Quick Start

```bash
cd transformer-foundations
uv pip install -e .

# Run Training
python3 -m transformer_foundations.train --dataset_path data/train.bin

# Run Generation
python3 -m transformer_foundations.generate --prompt "Once upon a time"
```
