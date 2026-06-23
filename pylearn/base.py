class BaseEstimator:
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError("Each estimator must implement a fit method.")

    def predict(self, X):
        raise NotImplementedError("Each estimator must implement a predict method.")
