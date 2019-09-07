'''
2019/09/06
Return : Transition matix, Relevance score, etc...
'''

import numpy as np
from scipy.sparse import coo_matrix
import os


def transpose_idx(data):
    '''
    args : data = np.array with shape (N,3)
           ijv format matrix
    '''

    # -1 : set index start from zero
    tmp = data[:,0].copy() - 1
    data[:,0] = data[:,1] - 1
    data[:,1] = tmp
    return data

# Open transition.txt file
# 81433 by 81433
# Return
def read_transition_matrix():
    '''
    return (ijv format matrix, array of idx indicating position of zero col)
    '''

    data = np.loadtxt("./data/transition.txt").astype('float32')
    data = transpose_idx(data)
    N = int(data.max()) + 1
    count = np.full(N,0,dtype = np.int32)

    zero_idx = np.array([])
    for row in data:
        count[int(row[1])] +=1
    
    for i,row in enumerate(data):
        data[i,2] /= count[int(row[1])]
        
    for i,v in enumerate(count):
        if (v == 0):
            zero_idx = np.append(zero_idx,i).astype('int32')


    row = data[:,0].astype('int32')
    col = data[:,1].astype('int32')
    val = data[:,2]

    data = coo_matrix((val,(row,col)),shape=(N,N))
    return data,zero_idx,N
            


def calculate(A,zero_idx,N,alpha = 0.1,epsilon = 10**(-8)):
    '''
    arg
        A : Sparse matrix by scify coo_matrix
        zero_idx : indicating idx of zero column
        alpha : hyperparmeter. 0.1 ~ 0.2
        epsilon : criterion of convergence

    return :
        Pagerank (principal eigenvector of B_pr)
    '''

    p0 = np.array(N*[1/N]).reshape([N,1])
    r0 = np.array(N*[1/N]).reshape([N,1])

    k = 0
    while(1):
        r1 = (1-alpha)*((A.__mul__(coo_matrix(r0))).toarray() + r0[zero_idx].sum()/N) + alpha*p0
        if (np.linalg.norm((r1-r0),ord = 1) <= epsilon):
            break
        r0 = r1
        k += 1

    print(k)
    return r0,r1

A,zero_idx,N = read_transition_matrix() 
r0,r1 = calculate(A,zero_idx,N)
print(r1.sum())


    