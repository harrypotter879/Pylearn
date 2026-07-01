import numpy as np
from .base import BaseEstimator


class MSELoss:
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def gradient(self, X, y_true, y_pred):
        error = y_pred - y_true
        n = len(y_true)

        dw = (2 / n) * np.dot(X.T, error)
        db = (2 / n) * np.sum(error)
        return dw, db
    
class RidgeLoss:

    def __init__(self, alpha):
        self.alpha = alpha

    def loss(self, y_true, y_pred, weights):
        mse = np.mean((y_true - y_pred) ** 2)
        w = self.alpha * np.sum(weights ** 2)
        return mse + w

    def gradient(self, X, y_true, y_pred, weights):

        error = y_pred - y_true
        n = len(y_true)

        dw = (2 / n) * X.T @ error + 2 * self.alpha * weights
        db = (2 / n) * np.sum(error)

        return dw, db
    
class LinearModel:
    def predict(self, X, weights, intercept):
        return np.dot(X, weights) + intercept


class Batch_gradient_Descent(BaseEstimator):

    def __init__(self,learning_rate,epochs,loss_fn=MSELoss(), model=LinearModel()):

        self.weights = None
        self.intercept = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.model = model

    def fit(self,X_train,y_train):

        self.weights = np.ones(X_train.shape[1])
        self.intercept = 0

        X_train_arr = np.array(X_train)
        y_train_arr = np.array(y_train)

        n = len(y_train_arr)

        for _ in range(self.epochs):

            y_pred = self.model.predict(X_train_arr, self.weights, self.intercept)
            
            slope_weights, slope_intercept = self.loss_fn.gradient(X_train_arr, y_train_arr, y_pred,self.weights)

            self.weights -= self.learning_rate * slope_weights
            self.intercept -= self.learning_rate * slope_intercept

        return self

    def predict(self,X_test):

        if self.weights is None or self.intercept is None:
            raise AttributeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'predict'."
            )
        
        X_test_arr = np.array(X_test)
        return self.model.predict(X_test_arr, self.weights, self.intercept)
    
    def fit_predict(self,X_train,y_train,X_test):

        self.fit(X_train,y_train)
        return self.predict(X_test)
    
class Stocastic_Gradient_Descent(BaseEstimator):

    def __init__(self,learning_rate,epochs):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.intercept = None

    def fit(self,X_train,y_train):

        X = np.array(X_train)
        y = np.array(y_train)

        self.weights = np.ones(X.shape[1])
        self.intercept = 0

        n = len(y)

        for _ in range(self.epochs):
            for _ in range(n):

                idx = np.random.randint(0,n)

                X_idx = X[idx]
                y_idx = y[idx]

                y_pred = np.dot(X_idx,self.weights) + self.intercept
                error = y_pred - y_idx 

                slope_weights = 2 * error * X_idx
                slope_intercept = 2 * error

                self.weights -= self.learning_rate * slope_weights
                self.intercept -= self.learning_rate * slope_intercept

        return self

    def predict(self, X_test):

        if self.weights is None or self.intercept is None:
            raise AttributeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'predict'."
            )
        
        X = np.array(X_test)
        return np.dot(X,self.weights) + self.intercept
    
    def fit_predict(self,X_train,y_train,X_test):
        
        self.fit(X_train,y_train)
        return self.predict(X_test)

class MiniBatch_Gradient_Descent(BaseEstimator):
    
    def __init__(self, learning_rate, epochs, batch_size):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.weights = None
        self.intercept = None

    def fit(self,X_train,y_train):

        X = np.array(X_train)
        y = np.array(y_train)

        self.weights = np.ones(X.shape[1])
        self.intercept = 0

        n = len(X)

        for _ in range(self.epochs):

            indices = np.random.permutation(n)
            X_suffled = X[indices]
            y_suffled = y[indices]

            for i in range(0,n,self.batch_size):

                X_batch = X_suffled[i:i+self.batch_size]
                y_batch = y_suffled[i:i+self.batch_size]

                y_pred = np.dot(X_batch, self.weights) + self.intercept
                error = y_pred - y_batch

                slope_weights = (2 / len(X_batch)) * (X_batch.T @ error)
                slope_intercept = (2 / len(X_batch)) * np.sum(error)

                self.weights -= self.learning_rate * slope_weights
                self.intercept -= self.learning_rate * slope_intercept

        return self
    
    def predict(self,X_test):

        if self.weights is None or self.intercept is None:
            raise AttributeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'predict'."
            )
        
        X_test_arr = np.array(X_test)
        return np.dot(X_test_arr,self.weights) + self.intercept

    def fit_predict(self,X_train,y_train,X_test):
        
        self.fit(X_train,y_train)
        return self.predict(X_test)



