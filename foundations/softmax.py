import numpy as np
import math
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        pass
        n = z.size
        x = z.max()
        total = 0
        for i in range(n):
            total += math.exp(z[i]-x)
            z[i] = math.exp(z[i]-x)
        for i in range(n):
            z[i] = round(z[i]/total,4)
        return z


