import numpy as np
from utils import *
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm

trainpath = '../data/train.csv'
devpath = '../data/dev.csv'
outpath = '../eval/dev_item_preds.csv'

def calculate(X,topK,score,method = "mean"):
    preds = []
    with open(devpath) as infile:
        for line in infile:
            mov_id, u_id= line.split(',')
            mov_id, u_id = int(mov_id), int(u_id)
            idx = np.setdiff1d(topK[mov_id],mov_id)  
            if method == "mean":
                predict = X[:,idx].mean(1)
            else:
                mov_id = np.ones_like(idx)*mov_id
                weighted = (score[mov_id,idx]/np.sum(score[mov_id,idx])).reshape([1,-1])
                predict = (X[:,idx].multiply(weighted)).sum(axis = 1)
                
            preds.append(str((predict[u_id,0]+3)))  
            
    with open(outpath,'w') as file:
        for line in preds:
            file.write(line)
            file.write('\n')
            
def kNN(score,k):
    '''
    Args :
        score = (train users, query users) csr matrix
        k = using k nearest neighbr
    Return :
        sort by the score matrix along with axis = 0, then
        return (users, top k users) matrix
    '''
    score = score.toarray()
    # for deleting same user
    
    #np.fill_diagonal(score,0)
    topK = (-score).argsort(axis = 0,kind = 'stable')
    return score,topK[:k+1].T

def DotProduct(X,query):
    '''
    Args : 
        X = (train user,movie) csr matrix
        query = (movie,query user) csr matrix
        
    Return : dot product with other users
    '''
    return X*query

args = load_review_data_matrix(trainpath)
X = args.X

itemSimilarity = DotProduct(X.transpose(),X)
score,topK = kNN(itemSimilarity,10)
calculate(X,topK,score,"weighted")
