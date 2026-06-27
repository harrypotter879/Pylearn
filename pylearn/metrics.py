import numpy as np
from .base import BaseEstimator

class Regression_Metrics(BaseEstimator):

    @staticmethod
    def MAE(y_test, y_pred):

        return np.mean(np.abs(y_test - y_pred))

    @staticmethod
    def MSE(y_test, y_pred):

        return np.mean((y_test - y_pred) ** 2)

    @staticmethod
    def RMSE(y_test, y_pred):

        return np.sqrt(Regression_Metrics.MSE(y_test, y_pred))

    @staticmethod
    def R2_Score(y_test, y_pred):

        SSr = np.sum((y_test - y_pred) ** 2)
        SSt = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (SSr / SSt)

    @staticmethod
    def Adjusted_R2(y_test, y_pred, X_train):

        r2 = Regression_Metrics.R2_Score(y_test, y_pred)
        n = len(y_test)

        has_intercept = any(np.all(X_train[:, i] == 1) for i in range(X_train.shape[1]))
        k = X_train.shape[1] if has_intercept else X_train.shape[1] + 1

        return 1 - (1 - r2) * (n - 1) / (n - k)

    