"""
cnn + LSTM for 1-d Multi-channel signal data, pytorch
based on 
https://github.com/hsd1503/resnet1d.git

changed some details by Qiuhong Zhang, Nov 2026

"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)

class CRNN(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        n_classes: number of classes
        
    """

    def __init__(self, in_channels,n_len_seg, n_classes, device, verbose=False):
        super(CRNN, self).__init__()
        
        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        # self.out_channels = out_channels

        self.device = device
        self.verbose = verbose

        # (batch, channels, length)
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, 
                            out_channels=32, 
                            kernel_size=8, 
                            stride=2)
        self.conv2 = nn.Conv1d(32, 128, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv1d(16, 64, kernel_size=2, stride=2)
        # self.conv4 = nn.Conv1d(64, 64, kernel_size=2, stride=2)
        # self.conv5 = nn.Conv1d(64, 256, kernel_size=1, stride=1)
        # self.conv6 = nn.Conv1d(256, 256, kernel_size=1, stride=1)
        self.pool = nn.MaxPool1d(2)

        # (batch, seq, feature)
        self.rnn = nn.LSTM(input_size=(128), 
                            hidden_size=128, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=False)
        self.dense = nn.Linear(128, n_classes)
        
    def forward(self, x):

        self.n_channel, self.n_length = x.shape[-2], x.shape[-1]
        assert (self.n_length % self.n_len_seg == 0), "Input n_length should divided by n_len_seg"
        self.n_seg = self.n_length // self.n_len_seg

        out = x
        if self.verbose:
            print(out.shape)

        # (n_samples, n_channel, n_length) -> (n_samples, n_length, n_channel)
        out = out.permute(0,2,1)
        if self.verbose:
            print(out.shape)
        # (n_samples, n_length, n_channel) -> (n_samples*n_seg, n_len_seg, n_channel)
        out = out.reshape(-1, self.n_len_seg, self.n_channel)
        if self.verbose:
            print(out.shape)
        # (n_samples*n_seg, n_len_seg, n_channel) -> (n_samples*n_seg, n_channel, n_len_seg)
        out = out.permute(0,2,1)
        if self.verbose:
            print(out.shape)
        # cnn
        # out = self.cnn(out)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        # out = F.relu(self.conv3(out))
        # out = F.relu(self.conv4(out))
        # out = self.pool(out)
        # out = F.relu(self.conv5(out))
        # out = F.relu(self.conv6(out))
        # out = self.pool(out)  

        if self.verbose:
            print(out.shape)
        # global avg, (n_samples*n_seg, out_channels)
        out = out.mean(-1)
        if self.verbose:
            print(out.shape)
        out = out.reshape(-1, self.n_seg, 128)
        if self.verbose:
            print(out.shape)
        _, (out, _) = self.rnn(out)
        out = torch.squeeze(out, dim=0)
        if self.verbose:
            print(out.shape)
        out = self.dense(out)
        if self.verbose:
            print(out.shape)
        
        return out
