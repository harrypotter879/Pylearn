import numpy as np

from .base import BaseEstimator

class Linear_Reg(BaseEstimator):

    def __init__(self):

        self.weights = None
        self.intercept = None

    def fit(self,X_train,y_train):

        X = np.c_[np.ones(X_train.shape[0]),X_train]
        Beta = np.linalg.solve(X.T @ X , X.T @ y_train)

        self.weights = Beta[1:]
        self.intercept = Beta[0]

        print(f'Weights : {self.weights}')
        print(f'Intercepts : {self.intercept}')

    def predict(self,X_test):

        return self.intercept +  X_test @ self.weights 
    
    def fit_predict(self,X_train,y_train,X_test):

        self.fit(X_train,y_train)
        return self.predict(X_test)