import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
from tqdm.auto import tqdm

from CNNBaseline import CNNBaseline
from RNNBaseline import LSTMBaseline
from data_process import *
from train import train_epoch
from train import test_epoch

# make csv file for using torch text

# make_csv_file_from_rawtext()

def main(args):
        method = args.method
        pretrainEmbedding = args.pretrainEmbedding
        makeCSVfile = args.makeCSVfile

        if makeCSVfile:
            make_csv_file_from_rawtext()     

        tokenize = lambda x: x.split()
        TEXT = Field(sequential=True, use_vocab=True,tokenize=tokenize, lower=True,pad_first = True)
        LABEL = Field(sequential=False, use_vocab=False,pad_token=None,unk_token=None)

        tv_datafield = [("id",None),("text",TEXT),("label",LABEL)]
        train_data = TabularDataset(path='./data/train_log.csv', format='csv', 
                fields=[('text', TEXT), ('label', LABEL)],skip_header=True)
        test_data = TabularDataset(path='./data/test_log.csv', format='csv', 
                fields=[('text', TEXT), ('label', LABEL)],skip_header=True)

        if pretrainEmbedding:
            vectors = Vectors(name="./data/all.review.vec.txt", cache='./')
            TEXT.build_vocab(train_data,max_size = 10000, vectors = vectors)
        else:
            TEXT.build_vocab(train_data,max_size = 10000)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vocab_size = len(TEXT.vocab)

        if method =="RNN":
            model = LSTMBaseline(vocab_size)
        else:
            model = CNNBaseline(vocab_size)

        model = model.to(device)
        traindl, testdl = torchtext.data.BucketIterator.splits(datasets=(train_data, test_data), # specify train and validation Tabulardataset
                                                batch_sizes=(32,32),  # batch size of train and validation
                                                sort_key=lambda x: len(x.text), # on what attribute the text should be sorted
                                                device=device, # -1 mean cpu and 0 or None mean gpu
                                                sort_within_batch=True, 
                                                repeat=False)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        epochs = 100

        trainAccuracy = []
        testAccuracy = []
        trainLoss = []
        testLoss = []
        trainTime = 0 

        for epoch in range(1,epochs + 1):
                loss,acc = train_epoch(model,traindl,optimizer,criterion)
                trainLoss.append(loss)
                trainAccuracy.append(acc)

                loss,acc = test_epoch(model,testdl,optimizer,criterion)
                testLoss.append(loss)
                testAccuracy.append(acc)

        print ("train Accuracy :", trainAccuracy[-1].item())
        print ("test Accuracy :", testAccuracy[-1].item())

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description = "text classification")

        parser.add_argument('--method',default = "CNN",help = 'RNN or CNN (default : RNN)')
        parser.add_argument('--pretrainEmbedding', default = False, \
                             help = 'use pretrained word embedding vector (default : False)')
        parser.add_argument('--makeCSVfile',default = True, \
                             help = 'Make csv file for torch text (default : True)')

        args = parser.parse_args()

        main(args)