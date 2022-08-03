import os
import json
import math
import torch
import torch.nn as nn
import datetime
import h5py
import pandas as pd 


class PositionalEncoding(nn.Module):

    def __init__(self, d_e=128, max_len=80):
        super(PositionalEncoding, self).__init__()
        root="/Users/ayshahchan/Desktop/ESPACE/thesis/codes/thesis/data"
        country='AT_T33UWP'
        path = os.path.join(root, "HDF5s", "train", country+"_"+"train"+".h5")
        h5_file = h5py.File(path, 'r')
        regions =  list(h5_file.keys())
        region= regions[0]
        h5_file_path = os.path.join(root, "HDF5s", "train", country+"_train.h5")
        data = pd.read_hdf(h5_file_path, region)
        self.dates_json = data.columns
        # Instead of taking the position, the numbers of days since the first observation is used
        days = torch.zeros(max_len)
        date_0 = self.dates_json[0]
        date_0 = datetime.datetime.strptime(str(date_0), "%Y%m%d")
        days[0] = 0
        for i in range(max_len - 1):
            date = self.dates_json[i + 1]
            date = datetime.datetime.strptime(str(date), "%Y%m%d")
            days[i + 1] = (date - date_0).days
        days = days.unsqueeze(1)

        # Calculate the positional encoding p
        p = torch.zeros(max_len, d_e)
        div_term = torch.exp(torch.arange(0, d_e, 2).float() * (-math.log(1000.0) / d_e))
        p[:, 0::2] = torch.sin(days * div_term)
        p[:, 1::2] = torch.cos(days * div_term)
        p = p.unsqueeze(0)
        self.register_buffer('p', p)

    def forward(self, x):
        
        x = x + self.p
        return x



# pe = PositionalEncoding()
# x = torch.zeros(128, 24, 128)
# x = pe.forward(x)
# print(x)
# print("hi")
