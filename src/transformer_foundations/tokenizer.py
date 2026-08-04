import itertools
import time
import re 
import heapq
import regex
import pickle
import numpy as np
from tqdm import tqdm
from itertools import batched
from collections.abc import Iterable, Iterator
import sys
# Merges is a list of tuple(bytes, bytes)
# Vocab is a dict of int : bytes in utf-8   
class Tokenizer(): 
    def __init__(self, vocab, merges, special_tokens=None): 
        self.vocab = vocab 
        self.merges = merges 
        self.special_tokens = special_tokens
        if self.special_tokens:
            for st in self.special_tokens: 
                if st.encode("utf-8") not in self.vocab.values():
                    self.vocab[len(self.vocab)] = st
        
    @classmethod
    def from_files(cls, vocab_filepath,merges_filepath, special_tokens=None):
    

        # 2. READ data back from the binary file
        with open(vocab_filepath, "rb") as file:
            loaded_vocab = pickle.load(file)
        with open(merges_filepath, "rb") as file:
            loaded_merges = pickle.load(file)

        parsed_vocab = loaded_vocab["vocab"]
        parsed_merges= loaded_merges["merges"]

        

        return cls(
            vocab=parsed_vocab, 
            merges=parsed_merges, 
            special_tokens=special_tokens
        )

    def encode(self, text: str) -> list[int]: 
        def get_pre_tokens(): 
            pre_tokens = []
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            
            if self.special_tokens:
                sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
                escaped_tokens = [re.escape(token) for token in sorted_tokens]
                pattern = "(" + "|".join(escaped_tokens) +")"
                sp_split = re.split(pattern, text)
                for i, chunk in enumerate(sp_split):
                    if i%2 ==1: 
                        pre_tokens.append(chunk)
                        continue
                    words = regex.finditer(PAT, chunk)
                    for match in words: 

                        word = tuple(bytes([b]) for b in match.group().encode("utf-8"))
                        if word:
                                pre_tokens.append(word)
            else:  
                    words = regex.finditer(PAT, text)
                    for match in words: 

                        word = tuple(bytes([b]) for b in match.group().encode("utf-8"))
                        if word:
                                pre_tokens.append(word)
            return pre_tokens


    
        def merge(token: tuple[bytes, ...], merge_ranks: dict[tuple[bytes, bytes], int]) -> tuple[bytes, ...]:
            if len(token) < 2:
                return token

            nodes = list(token)
            prev_indices = [i - 1 for i in range(len(token))]
            next_indices = [i + 1 for i in range(len(token))]
            next_indices[-1] = -1 

            heap = []

            for i in range(len(token)-1):
                pair = (token[i], token[i+1])
                if pair in merge_ranks:
                    heapq.heappush(heap, (merge_ranks[pair] , i, i+1, nodes[i], nodes[i+1] ))
    
            while heap:
                rank, i, j, e_i, e_j= heapq.heappop(heap)
                if  e_j != nodes[j] or e_i != nodes[i]: 
                    continue
                nodes[i] = nodes[i] + nodes[j]
                nodes[j] =b''
                jump_over_j = next_indices[j]
                next_indices[i] = jump_over_j
                if jump_over_j !=-1: 
                    prev_indices[jump_over_j] = i
                prev_i = prev_indices[i]
                if prev_i !=-1: 
                    left_pair = (nodes[prev_i], nodes[i])
                    if left_pair in merge_ranks:
                    
                        heapq.heappush(heap, (merge_ranks[left_pair], prev_i, i, nodes[prev_i], nodes[i] ))
                if jump_over_j !=-1: 
                    right_pair = (nodes[i], nodes[jump_over_j])
                    if right_pair in merge_ranks:

                        heapq.heappush(heap, (merge_ranks[right_pair], i , jump_over_j, nodes[i], nodes[jump_over_j]))
            curr = 0
            result = []
            while curr!=-1: 
                result.append(nodes[curr])
                curr = next_indices[curr]
                

            return tuple(result)
        
        pre_tokens = get_pre_tokens()
        tokenized_output = []
        token_cache = {}
        
        merge_ranks = {pair: rank_idx for rank_idx, pair in enumerate(self.merges)}
        for token in pre_tokens: 
            
            if token not in token_cache :  
                if self.special_tokens is not None and token in self.special_tokens:
                    token_cache[token]  = (token.encode("utf-8"), )
                    tokenized_output.append(token_cache[token])

                    continue
                token_cache[token] = merge(token, merge_ranks) 
            tokenized_output.append(token_cache[token])
        vocab_str_to_int = {v: k for k, v in self.vocab.items()}
            
        ids = [vocab_str_to_int[token] for tokens in tokenized_output for token in tokens]
        
        return ids
    def encode_iterable(self, iterable: Iterable[str]) ->Iterator[int]:

        for text in iterable:
            processed_chunk = self.encode(text)
            yield from processed_chunk

    def decode(self, ids: list[int]) -> str : 
        bytized = [self.vocab.get(id) for id in ids]
        result = b"".join(bytized)
        return result.decode("utf-8", errors= "replace")




def sample_from_dataset(input_path: str, num_samples: int = 10) -> str:
    samples = []
    docs = []
    with open(input_path, "r", encoding ="utf-8") as f:
        i =0
        for line in f: 
            samples.append(line)
            if "<|endoftext|>" in line: 
                i +=1
                docs.append("".join(samples))
                samples =[]
                if i == num_samples: 
                    break
        if samples:
            if "<|endoftext|>" not in samples[-1]:
                samples.append("<|endoftext|>\n")
            docs.append("".join(samples))
        return "".join(docs)

def tokenizer_and_save_to_file(input_path: str, output_path: str):
    t = Tokenizer.from_files("vocab.pkl", "merges.pkl", ["<|endoftext|>"])
    with open(input_path, "r") as f: 
        text = f.read()
        
    tokens = t.encode_iterable([text])
    DTYPE = np.uint16
    CHUNK_SIZE = 1_000_000
    BUFFER_SIZE = 64*1024*1024
    with open(output_path, "wb", buffering = BUFFER_SIZE) as f:
        with tqdm(desc="Data Written", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
       
            while True: 
                chunk = list(itertools.islice(tokens, CHUNK_SIZE)) 
                if not chunk: 
                    break
                chunk_bytes = np.array(chunk , dtype = DTYPE).tobytes()
                f.write(chunk_bytes)
                bytes_written = len(chunk) * np.dtype(DTYPE).itemsize 
                pbar.update(bytes_written)


# text = sample_from_dataset("/Users/abdullah/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 500)

# t = Tokenizer.from_files("vocab.pkl", "merges.pkl", ["<|endoftext|>"])
# t1 = time.time()
# sys.getsizeof(text)
# len_e = len(t.encode(text))
# t2 = time.time()
# est_throuput =sys.getsizeof(text) / (t2-t1)
# print(f"throuput at 500 docs bytes / s {est_throuput}")
# print(f"compress ratio bytes / token: {len_e / len(text) }")
# print(f"estimated time for 825GB in hours: {(825e9 / est_throuput ) / (60 * 60) }")
# tokenizer_and_save_to_file("/Users/abdullah/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", "tokens_train.bin")
# token_array = np.fromfile("tokens_train.bin", dtype=np.uint16)
# token_array[0:10]