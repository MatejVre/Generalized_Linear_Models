import numpy as np

class LinearRegression():

    def __init__(self):
        self.betas = None
        self.hat_matrix = None
    
    def build(self, X, y):
        X = np.column_stack((X, np.ones(X.shape[0])))
        matr_for_both = np.linalg.inv((X.T@X))@X.T
        self.betas = matr_for_both@y
        self.hat_matrix = X@matr_for_both

    def predict(self, X):
        X = np.column_stack((X, np.ones(X.shape[0])))
        return X@self.betas
