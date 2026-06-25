import numpy as np
from .base import BaseEstimator

class Batch_gradient_Descent(BaseEstimator):

    def __init__(self,learning_rate,epochs):

        self.weights = None
        self.intercept = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self,X_train,y_train):

        self.weights = np.ones(X_train.shape[1])
        self.intercept = 0

        X_train_arr = np.array(X_train)
        y_train_arr = np.array(y_train)

        n = len(y_train_arr)

        for _ in range(self.epochs):

            y_pred = np.dot(X_train_arr,self.weights) + self.intercept
            loss = y_train_arr - y_pred

            slope_weights = (-2 / n) * (X_train_arr.T @ loss)
            slope_intercept = (-2 / n) * np.sum(loss)

            self.weights -= self.learning_rate * slope_weights
            self.intercept -= self.learning_rate * slope_intercept

    def predict(self,X_test):

        if self.coef_ is None or self.intercept_ is None:
            raise AttributeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'predict'."
            )
        
        X_test_arr = np.array(X_test)
        return np.dot(X_test_arr,self.weights) + self.intercept
    
    def fit_predict(self,X_train,y_train,X_test):

        self.fit(X_train,y_train)
        return self.predict(X_test)

