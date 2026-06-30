import numpy as np

from .base import BaseEstimator

class Linear_Reg(BaseEstimator):

    def __init__(self):

        self.weights = None
        self.intercept = None
        self.beta = None

    def fit(self,X_train,y_train):

        X_arr = np.c_[np.ones(X_train.shape[0]),X_train]
        y_arr = np.array(y_train)

        beta = np.linalg.pinv(X_arr) @ y_arr

        self.beta = beta
        self.weights = beta[1:]
        self.intercept = beta[0]

        print(f'Weights : {self.weights}')
        print(f'Intercepts : {self.intercept}')

        return self

    def predict(self,X_test):

        if self.beta is None:
            raise AttributeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'predict'."
            )

        return X_test @ self.weights + self.intercept
    
    def fit_predict(self,X_train,y_train,X_test):

        self.fit(X_train,y_train)
        return self.predict(X_test)
    
class Polynomial_Regression(BaseEstimator):

    def __init__(self,degree):

        self.degree = degree
        self.weights = None
        self.intercept = None
        self.beta = None

        if self.degree < 0:
           raise ValueError("degree must be >= 0")
    
    def Poly_Transform(self,X):

        X = np.asarray(X)

        return np.column_stack(
            [X**i for i in range(self.degree + 1)] # [1, X, X^2, ..., X^degree]
        )

    def fit(self,X_train,y_train):

        X_arr = self.Poly_Transform(X_train)
        y_arr = np.array(y_train)

        beta = np.linalg.pinv(X_arr) @ y_arr
        
        self.weights = beta[1:]
        self.intercept = beta[0]
        self.beta = beta

        return self
    
    def predict(self,X_test):

        if self.beta is None:
            raise AttributeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'predict'."
            )
        
        X_arr = self.Poly_Transform(X_test)
        return X_arr @ self.beta
    
    def fit_predict(self,X_train,y_train,X_test):

        self.fit(X_train,y_train)
        
        return self.predict(X_test)

class Ridge_Regression(BaseEstimator):

    def __init__(self,alpha):

        self.weights = None
        self.intercept = None
        self.alpha = alpha
        self.beta = None

    def fit(self,X_train,y_train):

        X = np.array(X_train)
        y = np.array(y_train)

        X = np.c_[np.ones(len(X)),X]
        I = np.eye(X.shape[1])
        I[0, 0] = 0 
        beta = np.linalg.pinv(X.T @ X + self.alpha * I) @ X.T @ y

        self.weights = beta[1:]
        self.intercept = beta[0]
        self.beta = beta

    def predict(self,X_test):

        if self.beta is None:
            raise AttributeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'predict'."
            )
        
        X = np.array(X_test)
        return self.intercept + X @ self.weights
    
    def fit_predict(self,X_train,y_train,X_test):

        self.fit(X_train,y_train)
        
        return self.predict(X_test)