import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    def __init__(self,vocab_size,filter_number = 100, emb_dim = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim) 
        self.conv1 = nn.Conv2d(1,filter_number,(3,emb_dim)) # (100,)
        self.relu = nn.ReLU()
        self.predictor = nn.Linear(filter_number, 1)
    def forward(self,seq):
        x = self.embedding(seq) # (batchSize,seqSize,embSize)
        x = torch.transpose(x, 0, 1)
        x = x.unsqueeze(1) # (batchSize,1,seqSize,embSize)
        x = self.relu(self.conv1(x)) # (batchSize,100,H,1)
        x = torch.squeeze(x,-1) # (batchSize,100,H)
        x = F.max_pool1d(x, x.size(2)) # (batchSize,100,1)
        x = x.squeeze()
        x = self.predictor(x)
        return x