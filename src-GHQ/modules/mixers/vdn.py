import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self):  # keep attr same with others
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        return th.sum(agent_qs, dim=2, keepdim=True)
