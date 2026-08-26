from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        res = []
        for i in numbers:
            num = str(i)
            start = 0
            result = []
            j = len(num)
            while j>start:
                x = num[start:j:1]
                if x in vocab:
                    result.append(x)
                    start = j
                    j = len(num)
                else:
                    if j==start+1:
                        result.append(x)
                        start = j
                        j = len(num)
                    else:
                        j -= 1
            res.append(result)
        return res


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        start = 0
        ans = 0
        j = len(text)
        while j>start:
            x = text[start:j:1]
            if x in vocab:
                start = j
                j = len(text)
                ans += 1
            else:
                if j==start+1:
                    ans += 1
                    start = j
                    j = len(text)
                else:
                    j -= 1
        return ans

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        t = text.split()
        wc = len(t)
        tc = wc-1
        for word in t:
            start = 0
            j = len(word)
            while j>start:
                x = word[start:j:1]
                if x in vocab:
                    tc += 1
                    start = j
                    j = len(word)
                else:
                    if j==start+1:
                        tc += 1
                        start = j
                        j = len(word)
                    else:
                        j -= 1
        return round(tc/wc,4)
