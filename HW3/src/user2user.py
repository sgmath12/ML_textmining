import numpy as np
from utils import *
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm
trainpath = '../data/train.csv'
devpath = '../data/dev.queries'

def kNN(score,k):
    '''
    Args :
        score = (train users, query users) csr matrix
        k = using k nearest neighbr
    Return :
        sort by the score matrix along with axis = 0, then
        return sorted train user index matrix, (top k users, query users) 
    '''
    uniqueQuery = set(score.indices)
    kNN = np.zeros([len(uniqueQuery),k])
    # for deleting same user
    score.setdiag(0)

    for queryUser,j in (enumerate(tqdm(uniqueQuery))):
        userScore = score.getcol(j)
        row = userScore.tocsc().indices
        X = np.vstack((row,userScore.data))
        i = (-X[1]).argsort()
        topUser = X[:,i]
        topUser = topUser[0,0:k]
        kNN[queryUser] = topUser
        
    print (kNN)


def userDotProduct(X,query):
    '''
    Args : 
        X = (train user,movie) csr matrix
        query = (movie,query user) csr matrix
        
    Return : dot product with other users
    '''
    return X*query

args = load_review_data_matrix(trainpath)
X = args.X

data,rows_cols = load_query_data(devpath)
query = csr_matrix((data, rows_cols)).transpose()
userSimilarity = userDotProduct(X,query)
kNN(userSimilarity,5)

