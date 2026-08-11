import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        x = torch.randn(fan_out,fan_in)*((2/(fan_in+fan_out))**(0.5))
        x = torch.round(x,decimals=4)
        return x.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        x = torch.randn(fan_out,fan_in)*((2/fan_in)**(0.5))
        x = torch.round(x,decimals=4)
        return x.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        ans = []
        weights = []

        if init_type=="xavier":
            y = torch.randn(hidden_dim,input_dim)*((2/(hidden_dim+input_dim))**(0.5))
            weights.append(y)
            for i in range(num_layers-1):
                y = torch.randn(hidden_dim,hidden_dim)*((1/(hidden_dim))**(0.5))
                weights.append(y)

        elif init_type=="kaiming":
            y = torch.randn(hidden_dim,input_dim)*((2/(input_dim))**(0.5))
            weights.append(y)
            for i in range(num_layers-1):
                y = torch.randn(hidden_dim,hidden_dim)*((2/(hidden_dim))**(0.5))
                weights.append(y)
        
        else:
            y = torch.randn(hidden_dim,input_dim)
            weights.append(y)
            for i in range(num_layers-1):
                y = torch.randn(hidden_dim,hidden_dim)
                weights.append(y)

        x = torch.randn(1,input_dim)
        for y in weights:
            x = torch.relu(x@(y.T))
            std = torch.std(x)
            ans.append(round(std.item(),2))
        
        return ans

