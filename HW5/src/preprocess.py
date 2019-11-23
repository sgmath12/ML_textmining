
import ndjson
import json
import numpy as np
import string
from string import digits
from collections import Counter
import multiprocessing
import pdb
import time
import re
import pickle
from scipy import sparse
# load from file-like objects


def load_stopwords(path ='./data/stopword.list' ):
    stopWords = []
    with open('./data/stopword.list') as f:
        for line in f:
            # remove '\n'
            stopWords.append(line[:-1])
    return stopWords

def load_split(stopWords, path = './data/yelp_reviews_train.json'):
    remove_punctuation = str.maketrans('','',string.punctuation)
    text = []
    label = []

    with open(path) as f:
        for line in f:
            label.append(int(json.loads(line)['stars']))
            line = json.loads(line)['text'].lower()
            line = line.translate(remove_punctuation)
            line = line.split()
            
            for word in line:
                if any(c.isdigit() for c in word):          
                    line.remove(word)
                if word in stopWords:
                    line.remove(word)

            text.append(line)
            
    return text,len(text),label

def make_dictionary_worker(document):
    CTFdictionary = Counter()
    DFdictionary = Counter()
    for line in document:
        for word in line:
            if any(c.isdigit() for c in word):
                continue
            else:
                CTFdictionary[word] += 1

        line = list(set(line))
        for word in line:
            if any(c.isdigit() for c in word):
                continue
            else:
                DFdictionary[word] += 1


    return CTFdictionary,DFdictionary

#720936 "good"

def make_dictionary():
    data,N,label = load_split()
    stopWords = load_stopwords()

    numCpu = multiprocessing.cpu_count()
    k = N//numCpu
    pool = multiprocessing.Pool(numCpu)
    argument = [data[i:i+k] for i in range(0,N,k)]
    result = pool.map(make_dictionary_worker,argument) 

    CTFdictionary = Counter()
    DFdictionary = Counter()

    for ctf,df in result:
        CTFdictionary += ctf
        DFdictionary += df

    
    CTFdictionary = sorted(CTFdictionary.items(), key=lambda x: x[1],reverse = True)[:2000]
    DFdictionary = sorted(DFdictionary.items(), key=lambda x: x[1],reverse = True)[:2000]
    return data,label,CTFdictionary,DFdictionary

def feature_vector(documents,dictionary):
    
    keys = list(map(lambda x : dictionary[x][0] , range(2000)))
    word2idx = dict(zip(keys,range(2000)))
    col = []
    row = []
    values = []

    for i,doc in enumerate(documents):
        counter = Counter()
        for word in doc:
            idx = (word2idx.get(word,0))
            if idx ==0 :
                continue
            counter[word] += 1
        
        for key,value in counter.items():
            row.append(i)
            col.append(word2idx[key])
            values.append(value)
        
            
    data = sparse.csr_matrix((values, (row, col)), shape=(len(documents), 2000))
    
    return data
                
def getdata():
    START_TIME = time.time()
    stopWords = load_stopwords()
    data,N,label = load_split(stopWords)
    END_TIME = time.time()
    pdb.set_trace()
    #data,label,CTFdict,DFdict = make_dictionary()
    
    '''
    with open("./data/data.pkl","wb") as f:
        pickle.dump(data,f)

    with open("./data/label.pkl","wb") as f:
        pickle.dump(label,f)

    with open("./data/CTFdict.pkl","wb") as f:
        pickle.dump(CTFdict,f)
    
    with open("./data/DFdict.pkl","wb") as f:
        pickle.dump(DFdict,f)
     

    
    with open("./data/data.pkl","rb") as f:
        data = pickle.load(f)

    with open("./data/label.pkl","rb") as f:
        label = pickle.load(f)

    with open("./data/CTFdict.pkl","rb") as f:
        CTFdict = pickle.load(f)

    with open("./data/DFdict.pkl","rb") as f:
        DFdict = pickle.load(f)

    
    X = feature_vector(data,CTFdict)
    y = label

    with open("./data/X.pkl","wb") as f:
        pickle.dump(X,f)

    with open("./data/y.pkl","wb") as f:
        pickle.dump(y,f)
    

'''
    with open("./data/X.pkl","rb") as f:
        X = pickle.load(f)


    with open("./data/y.pkl","rb") as f:
        label = np.array(pickle.load(f))


    C = max(label)
    N = X.shape[0]
    y = np.zeros([X.shape[0],C])

    y[np.arange(N),label-1] =1
    
    return X,y
    

if __name__ == "__main__":
    X,y = getdata()
    