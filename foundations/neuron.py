import numpy as np
import math
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        y = np.dot(x,w) + b
        if(activation=="sigmoid"):
            if y>=0:
                y = 1/(1+math.exp(-1*y))
            else:
                z = math.exp(y)
                y = z/(1+z)
        else:
            y = max(0.0,y)
        return round(y,5)
