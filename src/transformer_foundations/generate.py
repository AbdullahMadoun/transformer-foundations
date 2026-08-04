
import torch 
import json
from cs336_basics.Transformer import Transformer
from cs336_basics.Tokenizer import Tokenizer
import random
import sys
import time
import os

def topk_sampling(logits):
    indices = torch.topk(logits, k =5 ).indices.tolist()
    sampled_token = random.sample(indices, k=1)
    return sampled_token
def topp_sampling(logits): 
    indices = torch.topk(logits, k =5 ).indices.tolist()
    sampled_token = random.sample(indices, k=1)
    return sampled_token
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

    
