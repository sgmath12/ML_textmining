import torch
import torch.optim as optim 

import numpy as np
from utils import *
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm

trainpath = '../data/train.csv'
outpath = '../eval/dev_pmf_preds.csv'
# 0.0001
lambda_U = 0.001
lambda_V = 0.0001
learning_rate = 5*(1e-4)
latent_dimension = 5
momentum = 0.9
num_iter = 300

def read_golden():   
    golden_in = open('../eval/dev.golden', 'r')
    ratings = []
    while True:
        g_line = golden_in.readline()
        if len(g_line) == 0:
            break
        rating = float(g_line.strip())
        ratings.append(rating)
    return np.array(ratings)

def creterion(I,X,U,V,lambda_U,lambda_V):
    # return ((X > 0).double() * (X-torch.mm(U,V))**2).mean()
    return (I * (X-torch.mm(U,V))**2).sum() + lambda_U*torch.sqrt((U**2).sum()) \
        + lambda_V*torch.sqrt((V**2).sum())
        

def test(R,y):
    preds = []
    with open('../data/dev.csv') as infile:
        for line in infile:
            mov_id, u_id= line.split(',')
            mov_id, u_id = int(mov_id), int(u_id)
            preds.append(R[u_id,mov_id]+3)
              
    preds = np.array(preds)
    rmse = np.sqrt(np.sum((preds-y)**2)/len(preds))
    print (" RMSE :",rmse)

    return

def train(I,X,y,d,num_iter,lr,momentum,lambda_U,lambda_V):
    '''
    Args :
        X = user,movie matrix
    Return :
        latent matrix U,V
    '''
    N,M = X.shape
    X = X.toarray()
    a,b = X.min(),X.max()


    R = torch.tensor(X, dtype = torch.double)
    U = torch.rand([N,d],dtype = torch.double).uniform_(a,b)
    V = torch.rand([d,M],dtype = torch.double).uniform_(a,b)
    I = torch.tensor(I,dtype = torch.double) 
    # rescaling
    U.requires_grad = True
    V.requires_grad = True


    optimizer_1 = optim.SGD([U], lr = lr,momentum = momentum)
    optimizer_2 = optim.SGD([V], lr = lr,momentum = momentum)

    for i in range(num_iter):
        optimizer_1.zero_grad()
        loss_1 = creterion(I,R,U,V,lambda_U,lambda_V)
        loss_1.backward()
        optimizer_1.step()
        
        optimizer_2.zero_grad()
        loss_2 = creterion(I,R,U,V,lambda_U,lambda_V)
        loss_2.backward()
        optimizer_2.step()

        if i % 2 == 0:
            print ("loss", (loss_1 + loss_2).item(),end= "")
            test(torch.mm(U,V).detach().numpy(),y)

            

    return torch.mm(U,V).detach().numpy()

args = load_review_data_matrix(trainpath)
X = args.X
args = load_review_data_matrix(trainpath,normalize = 0)
I = args.X
I = (I>0).toarray()
y = read_golden()
R = train(I,X,y,latent_dimension,num_iter,learning_rate,momentum,lambda_U,lambda_V)



