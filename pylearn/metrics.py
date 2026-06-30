import numpy as np

class MAE:
    def cal(self, y_test, y_pred):
        return np.mean(np.abs(y_test - y_pred))

class MSE:
    def cal(self, y_test, y_pred):
        return np.mean((y_test - y_pred) ** 2)

class RMSE:
    def cal(self, y_test, y_pred):
        return np.sqrt(MSE().cal(y_test, y_pred))

class R2_Score:
    def cal(self, y_test, y_pred):
        SSr = np.sum((y_test - y_pred) ** 2)
        SSt = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (SSr / SSt)

class Adjusted_R2:
    def cal(self, y_test, y_pred, X_train):
        r2 = R2_Score().cal(y_test, y_pred)
        n = len(y_test)
        has_intercept = any(np.all(X_train[:, i] == 1) for i in range(X_train.shape[1]))
        k = X_train.shape[1] if has_intercept else X_train.shape[1] + 1
        return 1 - (1 - r2) * (n - 1) / (n - k)

    