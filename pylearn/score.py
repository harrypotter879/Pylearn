import numpy as np

class RegressionMetrics:

    def mae(self, y_test, y_pred):
        return np.mean(np.abs(y_test - y_pred))

    def mse(self, y_test, y_pred):
        return np.mean((y_test - y_pred) ** 2)

    def rmse(self, y_test, y_pred):
        return np.sqrt(self.mse(y_test, y_pred))

    def r2_score(self, y_test, y_pred):
        SSr = np.sum((y_test - y_pred) ** 2)
        SSt = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (SSr / SSt)

    def adjusted_r2(self, y_test, y_pred, X_train):
        r2 = self.r2_score(y_test, y_pred)
        n = len(y_test)
        has_intercept = any(np.all(X_train[:, i] == 1) for i in range(X_train.shape[1]))
        k = X_train.shape[1] if has_intercept else X_train.shape[1] + 1
        return 1 - (1 - r2) * (n - 1) / (n - k)

class Classification_Matrics:

    def classification_graph(y_test,y_pred):

        TP = np.sum((y_test == 1) & (y_pred == 1))
        TN = np.sum((y_test == 0) & (y_pred == 0))
        FP = np.sum((y_test == 0) & (y_pred == 1))
        FN = np.sum((y_test == 1) & (y_pred == 0))

        print(f"[TN: {TN}] [FP: {FP}]")
        print(f"[FN: {FN}] [TP: {TP}]")

    def Accuracy(y_test,y_pred):
        return np.mean(y_test == y_pred)
    
    def precision(y_test,y_pred):

        TP = np.sum((y_test == 1) & (y_pred == 1))
        PP = np.sum(y_pred == 1) # predicted_positives
        if PP > 0:
            return TP / PP
        else:
            return 0
        
    def recall(y_test,y_pred):

        TP = np.sum((y_test == 1) & (y_pred == 1))
        AP = np.sum(y_test == 1) # Actual Prositives

        if AP > 0:
            return TP/AP
        else:
            return 0
        
    def F1_score(self,y_test,y_pred):

        P = self.precision(y_test,y_pred)
        R = self.recall(y_test,y_pred)

        return (2 * P * R)/(P + R)

    