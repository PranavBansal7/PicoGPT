import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List
from torch.nn.utils.rnn import pad_sequence

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combined = [sentence.split() for sentence in (positive + negative)]
        upos = sorted(set(word for sentence in combined for word in sentence))
        tab = {word:i for i,word in enumerate(upos,start=1)}
        x = [torch.tensor([tab[word] for word in sentence]) for sentence in combined]
        x = pad_sequence(x,batch_first=True,padding_value=0)
        return x.float()


