# Transformer Foundations

Stanford CS336 Assignment 1, done as self-study in summer 2026: a decoder-only language
model and a byte-level BPE tokenizer written in plain PyTorch, with my own Linear,
Embedding, RMSNorm, attention, loss and AdamW in place of the `torch.nn` and
`torch.optim` versions. The GPU follow-up (Assignment 2) is in
[gpu-systems](https://github.com/AbdullahMadoun/gpu-systems).

## Where things are

| File | What it is |
| :--- | :--- |
| `src/transformer_foundations/transformer.py` | Linear and Embedding (truncated-normal init, `einops.einsum`), RMSNorm computed in FP32, SwiGLU, RoPE, causal multi-head self-attention with a fused QKV projection, pre-norm blocks and the LM head. |
| `src/transformer_foundations/model.py` | Re-exports the model classes. |
| `src/transformer_foundations/bpe.py` | BPE training: splits the corpus at document boundaries with `mmap`, counts pre-tokens in parallel (`ProcessPoolExecutor`), then merges pairs with a max-heap and a pair-to-word index. |
| `src/transformer_foundations/tokenizer.py` | Encoder that applies merges with a min-heap over a linked list of positions, O(N log K), and streams token IDs to a `uint16` file. |
| `src/transformer_foundations/train_scaffold.py` | Cross-entropy with log-sum-exp, AdamW, cosine learning-rate schedule with warm-up, gradient clipping, batching from `np.memmap`, checkpoints. |
| `src/transformer_foundations/train.py` | Training loop, logged to Weights & Biases. |
| `src/transformer_foundations/generate.py` | Interactive generation from a saved checkpoint (top-k sampling). |
| `tests/test_model.py` | Shape and sanity checks for the model, RMSNorm and SwiGLU. |

## Run

```bash
uv pip install -e .
pytest tests
```

`train.py` and `generate.py` are the scripts I ran on my own machine. They read
`config_light.json` (model, optimizer and schedule settings), a pre-tokenized `.bin`
file and the saved tokenizer files, none of which are in the repo, and `train.py` still
points at my local data path.
