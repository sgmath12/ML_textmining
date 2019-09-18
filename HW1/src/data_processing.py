import numpy as np
from scipy.sparse import csc_matrix
import os


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

def read_topic_distro(Rt = 1,filename = "user-topic-distro.txt"):
    '''
    Args : 
        Rt = TSPR vectors
        filename = "user-topic-distro.txt" or "query-topic-distro.txt"
    Return :
        p(t|q) probablity of topic given q
    '''
    with open("./data/"+filename) as f:
        content = f.readlines()
 
    content = [x.strip() for x in content]
    Probability_given_query = {}
    #topics = np.empty((0,12), float)
    
    for line in content:
        topic_weight = np.array([])
        line = line.split()
        for t in range(2,14):
            topic_weight = np.append(topic_weight,float(line[t].split(':')[1]))

        topic_weight = topic_weight.reshape(12,1)
        Probability_given_query[int(line[0]),int(line[1])] = topic_weight

    return Probability_given_query

def calculate_TSPR(Rt,Probability_given_query,key):
    '''
    Args :
        Rt = (doc,topic)
        Probability_given_query : (topics,1)
    Return :
        (docs,1)
    '''
    return Rt*(Probability_given_query[key])

def indri_key():
    filename = "user-topic-distro.txt"
    keys = []
    with open("./data/"+filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for line in content:
        line = line.split()
        keys.append((int(line[0]),int(line[1])))
    
    return keys


def read_indri_file(queryID,filename = None):
    ''' 
    Args :
        queryID : a,b is just a unique id given by homework file. 
        
    Return :
        return idx and score of retrieval doc
    '''
    queryID = str(queryID[0]) + '-' + str(queryID[1])
    filename = queryID +".results.txt"
    data = np.genfromtxt('./data/indri-lists/' + '2-1.results.txt',dtype='str')
    doc_idx = (data[:,2]).astype('int32')
    score = (data[:,4]).astype('float32').reshape(len(data),1)
    return doc_idx,score


def weighted_sum(PageRank,doc_idx,score,alpha = 0.8):
    result = alpha * PageRank[doc_idx] + (1-alpha)*score   
    return result

def make_all_docs_score(PageRank,filename = "GPR_NS.txt"):
    f = open("guru99.txt","w+")




def main():
    # keys = unique query ID.
    keys = indri_key()
    keys.sort()

    A,zero_idx,docs = read_transition_matrix() 
    GPR = calculate(A,zero_idx,docs)
    
    P_t,_,_ = read_transition_matrix(filename = "doc_topics.txt")
    RT = np.empty((docs,0),float)
    for topic in range(12):
        r = calculate(A,zero_idx,docs,beta = 0.1,P_t = P_t.getcol(topic))
        RT = np.hstack((RT,r))

    Probability_given_querytopic = read_topic_distro(RT,filename = "query-topic-distro.txt")
    Probability_given_usertopoic = read_topic_distro(RT,filename = "user-topic-distro.txt")
    
    QTSPR = {}
    PTSPR = {}

    for key in keys:
        QTSPR[key] = calculate_TSPR(RT,Probability_given_querytopic,key)
        PTSPR[key] = calculate_TSPR(RT,Probability_given_usertopoic,key)


    GPR_WS = {}
    QTSPR_WS = {}
    PTSPR_WS = {}

    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        GPR_WS[key] = weighted_sum(GPR,doc_idx,score)

    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        QTSPR_WS[key] = weighted_sum(QTSPR[key],doc_idx,score)

    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        PTSPR_WS[key] = weighted_sum(PTSPR[key],doc_idx,score)

    

main()