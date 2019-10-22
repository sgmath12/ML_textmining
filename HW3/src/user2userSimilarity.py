import numpy as np
from utils import *
from scipy.sparse import csc_matrix, csr_matrix

path = '../data/train.csv'

score,user,movie = load_raw_review_data(path)

print ("the number of times any movie was rated '1' :" , np.sum(score ==1))
print ("the number of times any movie was rated '3' :",np.sum(score ==3))
print ("the number of times any movie was rated '5' :",np.sum(score ==5))
print ("the average movie rating across all users and movies %.3f : "%np.mean(score))
print ("the number of movies rated of user 4321 : ",np.sum(user==4321))
print ("the number of times the user gave a '1' rating : ",np.sum(score[user == 4321]==1))
print ("the number of times the user gave a '3' rating :",np.sum(score[user == 4321]==3))
print ("the number of times the user gave a '5' rating :",np.sum(score[user == 4321]==5))
print ("the average movie rating for this user : %.3f"%np.mean(score[user == 4321]))
print ("the number of users rating this movie 3 : ",np.sum(movie == 3))
print("the number of times the user gave a '1' rating :",np.sum(score[movie==3]==1))
print("the number of times the user gave a '3' rating :",np.sum(score[movie==3]==3))
print("the number of times the user gave a '5' rating :",np.sum(score[movie==3]==5))
print ("the average rating of this movie : %.3f"%np.mean(score[movie==3]))
# the average movie rating for this user

def NN(productResult,k,method = "user"): 
    '''
    Args : 
        productResult : CSR matrix
                        one of the dotproduct or cosine similarity
    '''
    #print (type(productResult))
    if method == "user":
        row = productResult.tocsc().indices
    else:
        row = productResult.indices
    score = productResult.data

    
    X = np.vstack((row,score))
    i = (-X[1]).argsort()
    topUser = X[:,i]
    #start from 1 since 0 is just query user
    print (topUser[0,1:k+1])
    print (topUser[1,1:k+1])
 
def itemDotProduct(itemid,X):
    '''
    Args : userid : int, X : csc matrix
    Return : dot product with other users
    '''
    itemVector = (X.getcol(itemid)).T
    return (itemVector*X).tocsr()

def itemCosine(itemid,X):
    '''
    Args : userid : int, X : csr matrix
    Return : cosine with other users
    '''
    norm = np.sqrt(X.multiply(X).sum(0))
    # for avoiding divide by zero
    norm[norm==0] = 1

    X = X.multiply((1/norm))
    X = X.tocsr()

    itemVector = (X.getcol(itemid)).T
    
    return (itemVector*X).tocsr()


def userDotProduct(userid,X):
    '''
    Args : userid : int, X : csr matrix
    Return : dot product with other users
    '''
    userVector = (X.getrow(userid)).T
    return X*userVector

def userCosine(userid,X):
    '''
    Args : userid : int, X : csr matrix
    Return : cosine with other users
    '''
    
    norm = np.sqrt(X.multiply(X).sum(1))
    # for avoiding divide by zero
    norm[norm==0] = 1
    X = X.multiply((1/norm))
    X = X.tocsr()

    #X = X.multiply(csr_matrix(1/np.sqrt(X.multiply(X).sum(1))))
    userVector = (X.getrow(userid)).T
    return X*userVector



args = load_review_data_matrix(path)

userDot = userDotProduct(4421,args.X)
NN(userDot,5)
userCosine = userCosine(4421,args.X)
NN(userCosine,5)

args = load_review_data_matrix(path,matrix_func=csc_matrix)

itemDot = itemDotProduct(3,args.X)
NN(itemDot,5,method = "item")
itemCosine = itemCosine(3,args.X)
NN(itemCosine,5,method = "item")