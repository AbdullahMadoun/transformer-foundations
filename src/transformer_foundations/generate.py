
import torch 
import json
from transformer_foundations.transformer import Transformer
from transformer_foundations.tokenizer import Tokenizer
import sys
import time
import os

def topk_sampling(probs: torch.Tensor, k: int = 5) -> list[int]:
    """Draw one token from the k most likely, in proportion to their renormalized probabilities."""
    top = torch.topk(probs, k)
    choice = torch.multinomial(top.values / top.values.sum(), 1)
    return [top.indices[choice].item()]

def topp_sampling(probs: torch.Tensor, p: float = 0.9) -> list[int]:
    """Nucleus sampling: draw one token from the smallest set of tokens whose
    probabilities sum to at least p, in proportion to their renormalized probabilities.

    probs: 1-D tensor of next-token probabilities (already softmaxed).
    Returns [token_id], the same shape topk_sampling returns.
    """
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    # Keep a token if the mass before it is still below p, so the token that crosses p stays in.
    keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < p
    kept = sorted_probs[keep]
    choice = torch.multinomial(kept / kept.sum(), 1)
    return [sorted_ids[keep][choice].item()]
with open('config_light.json', 'r') as file:
        config = json.load(file)
transformer_param = config["transformer"]
transformer = Transformer(**transformer_param)
sd = torch.load("checkpoint_light.pt")
transformer.load_state_dict(sd["model"])
tokenizer = Tokenizer.from_files("vocab.pkl", "merges.pkl", ["<|endoftext|>"])
input_text = input("Enter your starting text: ")

output = tokenizer.encode(input_text)
logits = transformer(output) 
i = len(output)
max_tokens =60
k = 5
input_tokens = output
while i < max_tokens: 
    if len(input_tokens)>=127: 
        break
    logits = torch.softmax(logits[-1], dim =-1)
    sampled_token = topk_sampling(logits)
    sampled_text = tokenizer.decode(sampled_token)
    if sampled_text == "<|endoftext|>": 
        break
    input_tokens.append(sampled_token[0])
    input_text+=sampled_text
    logits = transformer(input_tokens)
    os.system('clear') 
    
    # Print the clean, updated generation status block
    print(f"Token Count: {i}/{max_tokens}")
    print("-" * 30)
    print(input_text.strip())
    
    i += 1  # Increment loop counter to prevent freeze
    time.sleep(0.5)

    
