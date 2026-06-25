import numpy as np
from .base import BaseEstimator

class Regression_Metrics(BaseEstimator):

    def __init__(self,y_test,y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    def MAE(self):
        
        mae = np.mean(np.abs(self.y_test - self.y_pred))
        return mae
    
    def MSE(self):

        mse = np.mean(self.y_test - self.y_pred)
        return mse
    
    
    
    