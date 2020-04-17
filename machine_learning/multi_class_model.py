import numpy as np
import pickle
import torch
import torch.nn as nn

class MultiClass(nn.Module):
    def __init__(self, n_in=4, n_out=3):
        super(MultiClass, self).__init__()
        self.n_in = n_in        
        self.n_out = n_out

        self.linear = nn.Sequential(
            nn.Linear(self.n_in, self.n_out, bias=True),
        )
        self.logprob = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.logprob(x)
        return x


