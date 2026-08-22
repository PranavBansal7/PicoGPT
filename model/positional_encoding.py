import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        pe = np.zeros((seq_len,d_model))
        pos = np.arange(seq_len).reshape(seq_len,1)
        div = 10000.0**(np.arange(0,d_model,2)/d_model)
        pe[:,0::2] = np.sin(pos/div)
        pe[:,1::2] = np.cos(pos/div)
        return np.round(pe,5)

