import torch.nn as nn
import torch.nn.functional as F

class MultiClass(nn.Module):
    def __init__(self, n_in=4, n_hid = 32, n_out=3):
        super(MultiClass, self).__init__()
        self.n_in = n_in        
        self.n_hid = n_hid
        self.n_out = n_out

        self.linear = nn.Sequential(
            nn.Linear(self.n_in, self.n_hid),
            nn.ReLU(),
            nn.Linear(self.n_hid, self.n_out)
        )
        self.logprob = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.logprob(x)
        return x


