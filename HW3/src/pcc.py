import numpy as np
from utils import *
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm

trainpath = '../data/train.csv'
devpath = '../data/dev.queries'
outpath = '../eval/dev_pcc_preds.csv'

def calculate(X,topUser,score,method = "mean"):
    
    preds = []
    with open('../data/dev.csv') as infile:
        for line in infile:
            mov_id, u_id= line.split(',')
            mov_id, u_id = int(mov_id), int(u_id)
            idx = np.setdiff1d(topUser[u_id],u_id)
            if method == "mean":
                predict = X[idx,:].mean(0)
            else:
                u_id = np.ones_like(idx)*u_id      
                weighted = (score[idx,u_id]/np.sum(abs(score[idx,u_id]))).reshape([-1,1])
                weighted = np.nan_to_num(weighted)
                predict = (X[idx].multiply(weighted)).sum(axis = 0)

            preds.append(str((predict[0,mov_id]+3)))   
            
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
        return score, (users, top k users) matrix
    '''
    topUser = (-score).argsort(axis = 0,kind = 'stable')

    return score,topUser[:k+1].T

def corrcoef(X,query):
    '''
    Args : 
        X = (train user,movie) csr matrix
        query = (movie,query user) csr matrix
        
    Return : 
        (user,user) correlation coeeficient matrix 
    '''
    X = X.toarray()
    return np.cov(X)



args = load_review_data_matrix(trainpath)
X = args.X
total_movies = X.shape[1]

data,rows_cols,users = load_query_data(devpath)
max_query_user = max(list(set(users)))
query = csr_matrix((data, rows_cols),shape = [max_query_user+1,total_movies]).transpose()

userSimilarity = corrcoef(X,query)
score,topUser = kNN(userSimilarity,10)
calculate(X,topUser,score,"mean")



 
