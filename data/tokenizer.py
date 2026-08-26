from typing import List
from collections import Counter

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        token = list(corpus)
        merge = []
        for _ in range(num_merges):
            freq = Counter()
            for i in range(len(token)-1):
                pair = (token[i],token[i+1])
                freq[pair] += 1
            best_pair = None
            for i in freq:
                if best_pair is None:
                    best_pair = i
                elif freq[best_pair]<freq[i]:
                    best_pair = i
                elif freq[best_pair]==freq[i] and i<best_pair:
                    best_pair = i
            new_tok = []
            if best_pair is None:
                break
            merge.append(list(best_pair))
            i = 0
            while i<len(token):
                if i+1< len(token):
                    pair = (token[i],token[i+1])
                    if pair==best_pair:
                        new_tok.append(token[i]+token[i+1])
                        i += 2
                    else:
                        new_tok.append(token[i])
                        i += 1
                else:
                    new_tok.append(token[i])
                    i += 1
            token = new_tok
        return merge
        
