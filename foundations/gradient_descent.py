class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        pass
        a = learning_rate
        n = iterations
        x = init
        for i in range(n):
            x = x - a*2*x
        return round(x,5)
            
