import heapq
import re 
import sys
import mmap
import regex
import pickle
from tqdm import tqdm
import os
from typing import BinaryIO
import itertools
from concurrent.futures import ProcessPoolExecutor
class RevTuple:
    def __init__(self, t):
        self.t = t
    def __lt__(self, other):
        return self.t > other.t 
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def process_chunk_worker(args) -> dict:
    input_path, (start, end), special_tokens = args
    with open(input_path, "rb") as f:
       with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            chunk = mm[start:end].decode("utf-8", errors="ignore")
  
    counts = {}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if special_tokens:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        pattern = "(" + "|".join(escaped_tokens) +")"
        sp_split = re.split(pattern, chunk)
        for i, text_chunk in enumerate(sp_split):
            if i%2 ==1: 
                continue
            words = regex.finditer(PAT, text_chunk)
            for match in words: 
                word = match.group()
                counts[word] = counts.get(word, 0) + 1
    else:
        words = regex.finditer(PAT, chunk)
        for match in words: 
            word = match.group()
            counts[word] = counts.get(word, 0) + 1


    return counts

def run_parallel_pre_tokens(input_path: str, special_tokens: list[str])->dict: 
    with open(input_path, "rb") as f:
        num_workers = 8
        num_chunks = 80
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        boundaries = find_chunk_boundaries(f, num_chunks, split_token)
        
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        
    boundaries = [(start,end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    worker_tasks = [(input_path, pair, special_tokens) for pair in boundaries]
    
    global_counts = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_chunk_worker, worker_tasks)
        
        for chunk_counts in results:
            for word, count in chunk_counts.items():
                global_counts[word] = global_counts.get(word, 0) + count

    return global_counts
def train_bpe(input_path: str, save_file: str |  None,  vocab_size: int, special_tokens: list[str]): 
    merges = []

    # Split accross special tokens 
    # Construct search patter
    word_counts_str = run_parallel_pre_tokens(input_path, special_tokens)
    print("Got em pre_tokens")
    
    word_counts = {}
    word_splits = {}
    for w_str, count in word_counts_str.items(): 
        word = tuple(bytes([b]) for b in w_str.encode("utf-8"))
        word_splits[word] = word
        word_counts[word] = count
        
    pair_freq ={}
    pair_to_word = {}
    for word, count in word_counts.items(): 
            for i in range(len(word)-1):

                pair = word_splits[word][i: i+2]
                if pair not in pair_to_word:
                    pair_to_word[pair] = set([word])
                else: 
                    pair_to_word[pair].add(word)

                if pair not in pair_freq:
                    pair_freq[pair] =count
                else: 
                    pair_freq[pair] +=count
    

    heap_pairs = [(-count, RevTuple(key)) for key, count in pair_freq.items()]   
    heapq.heapify(heap_pairs) 
    vocab = {i: bytes([i]) for i in range(256)} 
    n_merges = vocab_size - len(vocab) - len(special_tokens)
    merges = []
    pbar = tqdm(range(n_merges))
    for i in pbar:
        while heap_pairs:
            count, pair_to_merge = heapq.heappop(heap_pairs)
            pair_to_merge = pair_to_merge.t
            count *=-1
            if count == pair_freq[pair_to_merge]: 
                break
        else: 
            break
        merged_pair = pair_to_merge[0] + pair_to_merge[1]
        merges.append(pair_to_merge)
        vocab[len(vocab)] = merged_pair
        for word in pair_to_word[pair_to_merge]:
            changes ={} 
            new_split = []
            i = 0 
            # Merge and update word_splits with correct formation
            word_freq = word_counts[word]
            current_split = word_splits[word]
            while i < len(current_split)-1: 
                pair = current_split[i: i +2]
                changes[pair] = changes.get(pair, 0) - word_freq
                i +=1
            i=0
            while i < len(current_split): 
                pair = current_split[i: i +2]
                if tuple(pair) == pair_to_merge: 
                    new_split.append(merged_pair)
                    i = i + 2
                else:
                    new_split.append(current_split[i])  
                    i = i + 1 

            word_splits[word] = tuple(new_split)
            # Update the pair_freq
            for i in range(len(new_split)-1): 
                pair = tuple(new_split[i : i+2])
                if pair not in pair_to_word:
                    pair_to_word[pair] = set([word])
                else: 
                    pair_to_word[pair].add(word)
                changes[pair] = changes.get(pair, 0) + word_freq
            for pair, change in changes.items(): 
                if change !=0: 
                    pair_freq[pair] = pair_freq.get(pair, 0 )  + change
                    if pair_freq[pair] > 0: 
                        heapq.heappush(heap_pairs, (- pair_freq[pair], RevTuple(pair)))

                
    
    for token in special_tokens: 
        vocab[len(vocab)] = token.encode("utf-8")
        
    if save_file =="True": 
        with open("merges.pkl", "wb") as file:
            data = {
                "merges": merges, 
            }
            pickle.dump(data, file)

        with open("vocab.pkl", "wb") as file:
            data = {
                "vocab": vocab, 
            }
            pickle.dump(data, file)
    return vocab, merges  
def main(args): 
    input_path = args[0]
    save_file = args[1]
    vocab, merges = train_bpe(
        input_path,
        save_file,
        10000, 
        ["<|endoftext|>"]
    )
    print(f"\nBPE execution complete. Final Vocab Size: {len(vocab)}")
if __name__ == "__main__":
    main(sys.argv[1:])
   