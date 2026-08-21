import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        act = []
        def hook_fn(module,inp,out):
            act.append(out.detach())
        handle = []
        for layer in model.modules():
            if isinstance(layer,nn.Linear):
                handle.append(layer.register_forward_hook(hook_fn))
        with torch.no_grad():
            model(x)
        for h in handle:
            h.remove()
        stats = []
        for y in act:
            d = (y<=0).all(dim=0)
            stats.append({
                "mean" : round(y.mean().item(),4),
                "std" : round(y.std().item(),4),
                "dead_fraction" : round(d.float().mean().item(),4),
            })
        return stats
        
    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        pred = model(x)
        loss = nn.MSELoss()(pred,y)
        loss.backward()
        stats = []
        for layer in model.modules():
            if isinstance(layer,nn.Linear):
                grad = layer.weight.grad
                stats.append({
                    "mean" : round(grad.mean().item(),4),
                    "std" : round(grad.std().item(),4),
                    "norm" : round(grad.norm().item(),4),
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        if any(act["dead_fraction"]>0.5 for act in activation_stats):
            return "dead_neurons"
        if any(grst["norm"]>1000 for grst in gradient_stats):
            return "exploding_gradients"
        if gradient_stats[len(gradient_stats)-1]["norm"] < 1e-5:
            return "vanishing_gradients"
        if any(act["std"]<0.1 for act in activation_stats):
            return "vanishing_gradients"
        if any(act["std"]>10.0 for act in activation_stats):
            return "exploding_gradients"
        return "healthy"
        
