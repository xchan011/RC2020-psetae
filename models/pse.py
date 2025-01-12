import torch
import torch.nn as nn
import numpy as np

from models.mlps import MLP1, MLP2


class SpatialEncoder(nn.Module):
    """
    First part of the presented architecture.
    Yields to a spatio-spectral embedding at time t
    """

    def __init__(self, device):
        super(SpatialEncoder, self).__init__()
        self.device = device
        self.mlp1 = MLP1()
        self.mlp2 = MLP2()

    def forward(self, x):  # x: [batch_size x set_size x seq_len x channels:10]
        # mask: [batch_size x seq_len x set_size]
        
        x = x.permute(0, 1, 2)    # from [batch_size x seq_len x channels:10 x set_size] to usual
        
        batch_size, seq_len, channels = x.shape
        
        x = x.permute(0, 1, 2).contiguous().view(-1, channels)  # [batch_size * seq_len x set_size x hidden_state:10]
        #mask = mask.contiguous().view(-1, set_size)            # [batch_size * seq_len x set_size]
        # geom = geom.unsqueeze(1).repeat(1, seq_len, 1).contiguous().view(-1, 4)
        
        #geom = geom.contiguous().view(-1, 4)
        mlp1_output = self.mlp1(x)                              # [batch_size * seq_len x set_size x hidden_state:64]
        
        #mlp1_output = mlp1_output.permute(1,0)              # [hidden_state:64 x batch_size * seq_len x set_size]
        #masked_output = mlp1_output * mask  # From Garnot paper # [hidden_state:64 x batch_size * seq_len x set_size]
        #masked_mean = torch.mean(masked_output, dim=-1)         # [hidden_state:64 x batch_size * seq_len]
        #mean_output = torch.mean(mlp1_output, dim=-1)
        #mean_output = mean_output.permute(1, 0) 
        #masked_mean = masked_mean.permute(1, 0)                 # [batch_size * seq_len x hidden_state:64]
        #std_output = torch.std(mlp1_output, dim=-1)
        #std_output = std_output.permute(1, 0) 
        #masekd_std = torch.std(masked_output, dim=-1)           # [hidden_state:64 x batch_size * seq_len]
        #masekd_std = masekd_std.permute(1, 0)                   # [batch_size * seq_len x hidden_state:64]

        #pooled = torch.cat((mlp1_output), dim=1)  # [batch_size * seq_len x hidden_state:132]
        pooled = mlp1_output.contiguous().view(batch_size, seq_len, -1)  # [batch_size x seq_len x hidden_state:132]
        pooled = pooled.type('torch.FloatTensor')
        mlp2_output = self.mlp2(pooled)                             # [batch_size x seq_len x hidden_state:128]
        return mlp2_output
