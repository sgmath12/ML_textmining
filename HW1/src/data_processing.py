import numpy as np
from scipy.sparse import csc_matrix
import os

'''
def transpose_idx(data):
    
    args : data = np.array with shape (N,3)
           ijv format matrix
    

    # -1 : set index start from zero
    tmp = data[:,0].copy() - 1
    data[:,0] = data[:,1] - 1
    data[:,1] = tmp
    return data
'''

def read_transition_matrix(filename = "transition.txt"):
    '''
    Return :
        (ijv format matrix, array of idx indicating position of zero col)
    '''
    data = np.loadtxt("./data/" +filename).astype('float32')
    #data = transpose_idx(data)
    docs = int(data.max()) 

    # theses values will only use in reading doc_topics.txt
    topics = int(max(data[:,1]))
    length = len(data)

    

    if data.shape[1] ==3:
        row = data[:,1].astype('int32') - 1
        col = data[:,0].astype('int32') - 1
        val = data[:,2]
    else:
        row = data[:,0].astype('int32') - 1
        col = data[:,1].astype('int32') - 1
        val = np.array([1]*length)

    zero_idx = np.setdiff1d(np.arange(docs),np.unique(col))
    if data.shape[1] == 3:
        data = csc_matrix((val,(row,col)),shape=(docs,docs))
    else:
        data = csc_matrix((val,(row,col)),shape=(docs,topics))
    

    row_sum = np.array(data.sum(axis = 0))
    # for avoiding zero divide
    row_sum[row_sum==0] = 1  
    data = data.multiply((1/row_sum))
    
    return data,zero_idx,docs
            
def calculate(A,zero_idx,docs,alpha = 0.8,beta = 0.0,epsilon = 10**(-8),P_t = None):
    '''
    arg
        A : Sparse matrix by scify coo_matrix
        zero_idx : indicating idx of zero column
        alpha : hyperparmeter : 0.1 ~ 0.2
        epsilon : criterion of convergence
    return :
        Pagerank (principal eigenvector of B_pr)
    '''

    
    p0 = r0 = np.array(docs*[1/docs]).reshape([docs,1])

    if P_t == None:
        P_t = 0
    gamma = 1 - alpha - beta


    while(1):

        r1 = (alpha)*((A.__mul__((r0))) + r0[zero_idx].sum()/docs) \
            + (beta)*P_t \
            + (gamma)*p0

        # L1-norm
        if (np.linalg.norm((r1-r0),ord = 1) <= epsilon):
            break
        r0 = r1
    return r1

def weighted_sum(rt = 1,filename = "query-topic-distro.txt"):
    '''
    Args : 
        rt = TSPR vectors
    Return :
        rq = TSPR vectors given a query q
        Here, there are total 38 queries, So this function returns (docs,38) np.array
    '''
    with open("./data/"+filename) as f:
        content = f.readlines()
 
    content = [x.strip() for x in content]
    topics = np.empty((0,12), float)
    
    for line in content:
        topic_weight = np.array([])
        line = line.split()
        for t in range(2,14):
            topic_weight = np.append(topic_weight,float(line[t].split(':')[1]))

        topic_weight = topic_weight.reshape(1,12)
        topics = np.append(topics, topic_weight, axis=0)

    topics = topics.transpose()
    print (rt.shape)
    print (topics.shape)

def read_indri_file(filename = None):
    data = np.genfromtxt('./data/indri-lists/' + '2-1.results.txt',dtype='str')
    data_2 = np.genfromtxt('./data/query-topic-distro.txt',dtype = 'str')
   # print (data[:,[2,4]])
    print (data_2[:,2:])


'''
A,zero_idx,docs = read_transition_matrix() 
r1 = calculate(A,zero_idx,docs)


P_t,_,_ = read_transition_matrix(filename = "doc_topics.txt")

rt = np.empty((docs,0),float)
for topic in range(12):
    r = calculate(A,zero_idx,docs,beta = 0.1,P_t = P_t.getcol(topic))
    rt = np.hstack((rt,r))
   

weighted_sum(rt)
'''
read_indri_file()



