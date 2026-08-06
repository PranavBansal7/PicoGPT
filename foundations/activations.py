import numpy as np
import math
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        pass
        n = z.size
        for i in range(n):
            z[i] = round(1/(1+(math.e)**(-1*z[i])),5)
        return z

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        pass
        n = z.size
        for i in range(n):
            z[i] = max(0,z[i])
        return z
