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
    data = np.genfromtxt('./data/indri-lists/' + queryID+ '.results.txt',dtype='str')
    doc_idx = (data[:,2]).astype('int32')
    score = (data[:,4]).astype('float32').reshape(len(data),1)
    return doc_idx,score


def weighted_sum(PageRank,doc_idx,score,alpha = 0.8):
    '''
    Args :
        PageRank : GPR or QTSPR or PTSPR (vectors)
    Return :
        Weighted sum of PageRank and retrieval score
    '''
    N = len(doc_idx)

    result = doc_idx.reshape(N,1)
    result = np.hstack((result,alpha * PageRank[doc_idx] + (1-alpha)*score))
    # sort
    result = result[result[:,1].argsort()]
    return result[::-1]

def custom_weighted_sum(PageRank,doc_idx,score,alpha = 0.4):
    cumulated_sum = 0
    threshold = 0.000001
    N = len(doc_idx)
    result = doc_idx.reshape(N,1)
    k = -1
    
    for i in range(1,N-1):
        cumulated_sum += abs(score[i+1] - score[i])
        score[i] = [k]
        if cumulated_sum > threshold:
            cumulated_sum = 0
            k -= 1
    score[N-1] = [k]
    
    result = np.hstack((result,alpha * PageRank[doc_idx] + (1-alpha)*score))
    result = result[result[:,1].argsort()]


    return result

def make_all_queries_score(PageRank,keys,filename = "GPR_WS.txt"):
    # 10-1 Q0 clueweb09-enwp03-35-1378 1 16 run-1
    result = []
    for key in keys:
        for i,value in enumerate(PageRank[key]):
            result.append((str(key[0])+'-'+str(key[1]) + " Q0" +' '+str(int(value[0])) +' '+\
                 ' '+ str(i+1) +' '+ ' ' + str(value[1]) \
                + " run-1"))
    
    
    with open(filename, 'w') as f:
        for item in result:
            f.write("%s\n" % item)
    



def main():
    # keys = unique query ID.
    keys = indri_key()
    keys.sort()

    A,zero_idx,docs = read_transition_matrix() 
    GPR_tmp = calculate(A,zero_idx,docs)
    
    P_t,_,_ = read_transition_matrix(filename = "doc_topics.txt")
    RT = np.empty((docs,0),float)
    for topic in range(12):
        r = calculate(A,zero_idx,docs,beta = 0.1,P_t = P_t.getcol(topic))
        RT = np.hstack((RT,r))

    Probability_given_querytopic = read_topic_distro(RT,filename = "query-topic-distro.txt")
    Probability_given_usertopoic = read_topic_distro(RT,filename = "user-topic-distro.txt")
    
    GPR = {}
    QTSPR = {}
    PTSPR = {}

    for key in keys:
        GPR[key] = GPR_tmp
        QTSPR[key] = calculate_TSPR(RT,Probability_given_querytopic,key)
        PTSPR[key] = calculate_TSPR(RT,Probability_given_usertopoic,key)



    GPR_WS  = {}
    GPR_CM = {}
    QTSPR_WS = {}
    QTSPR_CM = {}
    PTSPR_WS = {}

    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        GPR_WS[key] = weighted_sum(GPR[key],doc_idx,score)


    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        GPR_CM[key] = custom_weighted_sum(GPR[key],doc_idx,score)

    print (GPR[key].shape)
    print (GPR_CM[key].shape)
    
    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        QTSPR_WS[key] = weighted_sum(QTSPR[key],doc_idx,score).squeeze()
        

    print (QTSPR[key].shape)
    print (QTSPR_WS[key].shape)

    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        QTSPR_CM[key] = custom_weighted_sum(QTSPR[key],doc_idx,score)

    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        PTSPR_WS[key] = weighted_sum(PTSPR[key],doc_idx,score)

   
    make_all_queries_score(QTSPR_CM,keys = keys,filename = "test_x.txt")

main()