import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        x = raw_dataset.split()
        a = []
        b = []
        for i in range(batch_size):
            j = torch.randint(0,len(x)-context_length,(1,)).item()
            p = x[j:j+context_length]
            q = x[j+1:j+context_length+1]
            a.append(p)
            b.append(q)
        return (a,b)
