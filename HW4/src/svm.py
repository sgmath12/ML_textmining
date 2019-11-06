import numpy as np
import sklearn
from conjugateGradient import conjugateGradient as cg

class SVM(object):

    def __init__(self,batch_size,features):
        self.C =  0.007812
        self.w = np.zeros(shape=(features,1))
        #self.lambda_ = 3631.3203125
        self.lambda_ = self.C * batch_size

    def criterion(self,X,y):
        B = X.shape[0]
        tmp = 1 - (X*self.w) * y.reshape([-1,1]) # (N,1)
        I = (tmp > 0).astype('int')
        
        # (1/2)*np.linalg.norm(self.w)**2  \
        # (self.Lambda/N)*np.matmul(np.ones([1,N]),tmp*tmp*I) + (self.Lambda/N)*np.matmul(np.ones([1,N]),tmp*tmp*I)
        return (1/2)*np.linalg.norm(self.w)**2 + \
                 (self.C)*np.matmul(np.ones([1,B]),tmp*tmp*I)
            
    def gradient(self,X,y,num_iterations):
        #X = X.toarray()
        B = X.shape[0]

        tmp = 1 - np.matmul(X,self.w) * y # (N,1)
        I = (tmp > 0).astype('int')
        X_I = X*I
        
        #return self.w + 2*(0.0078125) * np.matmul(X_I.T,np.matmul(X_I,self.w) - y*I),I
        return self.w/num_iterations +  2*(self.lambda_/B)*np.matmul(X_I.T,np.matmul(X_I,self.w) - y*I),I

    def predict(self,X,y):
       
        preds = X*self.w
        preds = np.where(preds>0,1,-1)
        print (np.mean(preds==y.reshape([-1,1])))
                                            
    def train(self,X,y,lr,batch_size,epoches,method ="SGD"):
        total_training_cases = X.shape[0]
        num_iterations = total_training_cases//batch_size + 1
        beta = 0.000
         
        
        for epoch in range(epoches):
            total_loss = 0
            lr = lr/(1 + beta * epoch)
            totalI = []
            totaldw = []
            for i in range(num_iterations):
                start_idx = (i * batch_size) % total_training_cases
                X_batch = X[start_idx:start_idx + batch_size].toarray()
                y_batch = y[start_idx:start_idx + batch_size].reshape([-1,1])

                dw,I = self.gradient(X_batch,y_batch,num_iterations)

                if method =="SGD":
                    self.w = self.w - lr*dw
                else :
                    dw = dw.reshape([-1])  
                    I = I.squeeze()
                    I = np.argwhere(I>0).squeeze()
                    totalI.append(I)
                    totaldw.append(dw)
                     
            if method == "Newton":
                totaldw = sum(totaldw)
                totalI = np.concatenate(totalI)
                d,_ = cg(X,totalI,totaldw,3631.3203125)
                self.w = self.w + d.reshape([-1,1])
                
            total_loss = self.criterion(X,y)
            print (total_loss.squeeze())
           
            # total_loss += loss
            # print (0.0078125*(total_loss ).squeeze())
            
            
        #self.predict(X,y)
             


    
