import numpy as np
from scipy.sparse import csc_matrix
import time
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
    return np.array(Rt*(Probability_given_query[key]))

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
    doc_idx = (data[:,2]).astype('int32') -1 
    score = (data[:,4]).astype('float32').reshape(len(data),1)
    return doc_idx,score


def weighted_sum(PageRank,doc_idx,score,alpha = 0.99):
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
    

def custom_weighted_sum(PageRank,doc_idx,score,alpha = 0.1):
    cumulated_sum = 0
    threshold = 0.00001
    epsilon = 0.000005
    N = len(doc_idx)
    result = doc_idx.reshape(N,1)
    k = -1

    
    for i in range(0,N-1):
        cumulated_sum += abs(score[i+1] - score[i])
        score[i] = [-k]
        if cumulated_sum> threshold:
            cnt = 0
            k -= epsilon

    score[N-1] = [-k]

    result = np.hstack((result,alpha * PageRank[doc_idx] + (1-alpha)*score))
    result = result[result[:,1].argsort()]


    return result

def make_all_queries_score(PageRank,keys,filename = "GPR_WS.txt"):
    # 10-1 Q0 clueweb09-enwp03-35-1378 1 16 run-1
    result = []
    for key in keys:
        for i,value in enumerate(PageRank[key]):
            #print (value)
            result.append((str(key[0])+'-'+str(key[1]) + " Q0" +' '+str(int(value[0])+1) +' '+\
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
    start_time = time.time()
    GPR_tmp = calculate(A,zero_idx,docs)
    
    P_t,_,_ = read_transition_matrix(filename = "doc_topics.txt")
    RT_time_start = time.time()
    RT = np.empty((docs,0),float)
    for topic in range(12):
        r = calculate(A,zero_idx,docs,beta = 0.1,P_t = P_t.getcol(topic))
        RT = np.hstack((RT,r))
    RT_time_end = time.time()
    RT_time = RT_time_end - RT_time_start

    Probability_given_querytopic = read_topic_distro(RT,filename = "query-topic-distro.txt")
    Probability_given_usertopoic = read_topic_distro(RT,filename = "user-topic-distro.txt")
    
    GPR = {}
    QTSPR = {}
    PTSPR = {}

    #Pagerank time -> take average except GPR
    GPR_time_start = time.time()
    for key in keys:
        GPR[key] = calculate(A,zero_idx,docs)
    GPR_time_end = time.time()
    GPR_time = (GPR_time_end-GPR_time_start)/len(keys)

    QTSPR_time_start = time.time()
    for key in keys:
        QTSPR[key] = calculate_TSPR(RT,Probability_given_querytopic,key)
    QTSPR_time_end = time.time()
    QTSPR_time = (QTSPR_time_end-QTSPR_time_start)/len(keys) + RT_time

    PTSPR_time_start = time.time()
    for key in keys:
        PTSPR[key] = calculate_TSPR(RT,Probability_given_usertopoic,key)
    PTSPR_time_end = time.time()
    PTSPR_time = (PTSPR_time_end-PTSPR_time_start)/len(keys) + RT_time

    GPR_NS = {}
    GPR_WS  = {}
    GPR_CM = {}

    QTSPR_NS = {}
    QTSPR_WS = {}
    QTSPR_CM = {}

    PTSPR_NS = {}
    PTSPR_WS = {}
    PTSPR_CM = {}

    # retrieval time -> take average for queries
    GPR_NS_time_start = time.time()
    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        GPR_NS[key] = weighted_sum(GPR[key],doc_idx,score,alpha = 1.0)
    GPR_NS_time_end = time.time()
    GPR_NS_time = (GPR_NS_time_end-GPR_NS_time_start)/len(keys)
    
    GPR_WS_time_start = time.time()
    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        GPR_WS[key] = weighted_sum(GPR[key],doc_idx,score)
    GPR_WS_time_end = time.time()
    GPR_WS_time = (GPR_WS_time_end-GPR_WS_time_start)/len(keys)

    GPR_CM_time_start = time.time()
    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        GPR_CM[key] = custom_weighted_sum(GPR[key],doc_idx,score)
    GPR_CM_time_end = time.time()
    GPR_CM_time = (GPR_CM_time_end-GPR_CM_time_start)/len(keys)

    QTSPR_NS_time_start = time.time()
    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        QTSPR_NS[key] = weighted_sum(QTSPR[key],doc_idx,score,alpha = 1.0)
    QTSPR_NS_time_end = time.time()
    QTSPR_NS_time = (QTSPR_NS_time_end-QTSPR_NS_time_start)/len(keys)

    QTSPR_WS_time_start = time.time()
    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        QTSPR_WS[key] = weighted_sum(QTSPR[key],doc_idx,score)
    QTSPR_WS_time_end = time.time()
    QTSPR_WS_time = (QTSPR_WS_time_end-QTSPR_WS_time_start)/len(keys)

    QTSPR_CM_time_start = time.time()
    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        QTSPR_CM[key] = custom_weighted_sum(QTSPR[key],doc_idx,score)
    QTSPR_CM_time_end = time.time()
    QTSPR_CM_time = (QTSPR_CM_time_end-QTSPR_CM_time_start)/len(keys)

    PTSPR_NS_time_start = time.time()
    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        PTSPR_NS[key] = weighted_sum(PTSPR[key],doc_idx,score,alpha = 1.0)
    PTSPR_NS_time_end = time.time()
    PTSPR_NS_time = (PTSPR_NS_time_end-PTSPR_NS_time_start)/len(keys)

    PTSPR_WS_time_start = time.time()    
    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        PTSPR_WS[key] = weighted_sum(PTSPR[key],doc_idx,score)
    PTSPR_WS_time_end = time.time()
    PTSPR_WS_time = (PTSPR_WS_time_end-PTSPR_WS_time_start)/len(keys)


    PTSPR_CM_time_start = time.time()
    for key in keys:
        #key = str(key[0])+'-'+str(key[1])
        doc_idx,score = read_indri_file(key)
        PTSPR_CM[key] = custom_weighted_sum(PTSPR[key],doc_idx,score)
    PTSPR_CM_time_end = time.time()
    PTSPR_CM_time = (PTSPR_CM_time_end-PTSPR_CM_time_start)/len(keys)


    make_all_queries_score(GPR_NS,keys = keys,filename = "GPR-NS.txt")
    print("GPR_NS : %f sec for PageRank, %f sec for retrieval"%(GPR_time,GPR_NS_time))
    make_all_queries_score(GPR_WS,keys = keys,filename = "GPR-WS.txt")
    print("GPR_WS : %f sec for PageRank, %f sec for retrieval"%(GPR_time,GPR_WS_time))
    make_all_queries_score(GPR_CM,keys = keys,filename = "GPR-CM.txt")
    print("GPR_CM : %f sec for PageRank, %f sec for retrieval"%(GPR_time,GPR_CM_time))
    make_all_queries_score(QTSPR_NS,keys = keys,filename = "QTSPR-NS.txt")
    print("QTSPR_NS : %f sec for PageRank, %f sec for retrieval"%(QTSPR_time,QTSPR_NS_time))
    make_all_queries_score(QTSPR_WS,keys = keys,filename = "QTSPR-WS.txt")
    print("QTSPR_WS : %f sec for PageRank, %f sec for retrieval"%(QTSPR_time,QTSPR_WS_time))
    make_all_queries_score(QTSPR_CM,keys = keys,filename = "QTSPR-CM.txt")
    print("QTSPR_CM : %f sec for PageRank, %f sec for retrieval"%(QTSPR_time,QTSPR_CM_time))
    make_all_queries_score(PTSPR_NS,keys = keys,filename = "PTSPR-NS.txt")
    print("PTSPR_CM : %f sec for PageRank, %f sec for retrieval"%(PTSPR_time,PTSPR_NS_time))
    make_all_queries_score(PTSPR_WS,keys = keys,filename = "PTSPR-WS.txt")
    print("PTSPR_CM : %f sec for PageRank, %f sec for retrieval"%(PTSPR_time,PTSPR_WS_time))
    make_all_queries_score(PTSPR_CM,keys = keys,filename = "PTSPR-CM.txt")
    print("PTSPR_CM : %f sec for PageRank, %f sec for retrieval"%(PTSPR_time,PTSPR_CM_time))
    

main()