import torch
from torch import nn, Tensor

class MegaLoc(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")