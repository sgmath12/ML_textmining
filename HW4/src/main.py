import sys
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import conjugateGradient as cg
from svm import *


def main():
    # read the train file from first arugment
    train_file = sys.argv[1]

    # read the test file from second argument
    test_file = sys.argv[2]

    # You can use load_svmlight_file to load data from train_file and test_file
    X_train, y_train = load_svmlight_file(train_file)
    X_test, y_test = load_svmlight_file(test_file)

    # add bias
    ones = np.ones([X_train.shape[0],1])
    X_train = csr_matrix(hstack([X_train,ones]))

    ones = np.ones([X_test.shape[0],1])
    X_test = csr_matrix(hstack([X_test,ones]))
    
    batch_size = 5000
    model = SVM(batch_size,features=X_train.shape[1])
    model.train(X_train,y_train,0.001,batch_size,150,"Newton")

    model.predict(X_train,y_train)
    model.predict(X_test,y_test)
    # You can use cg.ConjugateGradient(X, I, grad, lambda_)
    

# Main entry point to the program
if __name__ == '__main__':
    main()
