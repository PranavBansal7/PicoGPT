import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        act = []
        def hook_fn(model,inp,out):
            act.append(out.detach())
        handles = []
        for layer in model.modules():
            if isinstance(layer,nn.ReLU):
                handles.append(layer.register_forward_hook(hook_fn))
        with torch.no_grad():
            model(x)
        for h in handles:
            h.remove()
        stats = []
        for a in act:
            d = (a==0).all(dim=0)
            stats.append(round(d.float().mean().item(),4))
        return stats

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if any(d>0.5 for d in dead_fractions):
            return "use_leaky_relu"
        if dead_fractions[0]>0.3:
            return "reinitialize"
        if all(x<y for x,y in zip(dead_fractions,dead_fractions[1:])) and dead_fractions[len(dead_fractions)-1]>0.1:
            return "reduce_learning_rate"
        return "healthy"
