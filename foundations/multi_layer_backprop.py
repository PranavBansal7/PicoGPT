import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        z1 = [sum(x[i]*W1[j][i] for i in range(len(x))) + b1[j]
              for j in range(len(W1))]
        for i in range(len(z1)):
            z1[i] = max(0,z1[i])
        y_hat = [sum(z1[i]*W2[j][i] for i in range(len(z1))) + b2[j] for j in range(len(W2))]
        loss = round(sum((y_hat[i]-y_true[i])**2/len(y_true) for i in range(len(y_true))),4)
        der_loss_w2 = [[round(2*(y_hat[i]-y_true[i])*z1[j]/len(y_true),4) for j in range (len(z1))] for i in range(len(y_true))] 
        der_loss_b2 = [round(2*(y_hat[i]-y_true[i])/len(y_true),4) for i in range(len(y_true))]
        der_loss_w1 = [[round(sum((2*(y_hat[j]-y_true[j])*W2[j][i]/len(y_true) for j in range(len(y_true))))*x[k] if z1[i]>0 else 0.0,4) for k in range(len(x))] for i in range(len(b1))]
        der_loss_b1 = [round(sum(2*(y_hat[j]-y_true[j])*W2[j][i]/len(y_true) if z1[i]>0 else 0.0 for j in range(len(y_true))),4) for i in range(len(b1))]
        return {
            "loss" : loss,
            "dW1" : der_loss_w1,
            "db1" : der_loss_b1,
            "dW2" : der_loss_w2,
            "db2" : der_loss_b2
        }

