import torch
import torch.nn as nn
import torch.nn.functional as F

# LoRA Linear Layer 
class LoRALinear(nn.Module):
    def __init__(self, in_features:int, out_features: int, rank: int, bias:bool, alpha: int=None):
        super().__init__()

        if alpha is None :
            alpha = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=False)

        if bias :
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.bias = None
        
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank))

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.lora_b)
    
    def forward(self, x: torch.tensor):
        result = nn.functional.linear(x, self.weight, bias=self.bias)
        result += ( x @ self.lora_a.T @ self.lora_b.T ) * self.scaling
        return result

