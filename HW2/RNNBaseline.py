import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBaseline(nn.Module):
    def __init__(self,vocab_size,hidden_dim = 100, emb_dim=100, num_linear=1):
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size, emb_dim) #seq length,Batch emb dimention
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1,:,:]
        preds = self.predictor(feature)
        return preds