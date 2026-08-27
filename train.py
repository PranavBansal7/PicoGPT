import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        loss = 0
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            x = []
            y = []
            for i in range(batch_size):
                s = torch.randint(0,len(data)-context_length,(1,)).item()
                a = data[s:s+context_length]
                b = data[s+1:s+context_length+1]
                x.append(a)
                y.append(b)
            x = torch.stack(x)
            y = torch.stack(y)
            logits = model(x)
            logits = logits.reshape(-1,logits.shape[-1])
            y = y.reshape(-1)
            loss = F.cross_entropy(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return round(loss.item(),4)
