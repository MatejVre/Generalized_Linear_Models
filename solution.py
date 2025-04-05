import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer



class MultinomialLogReg():

    def __init__(self):
        self.betas = None

    def build(self, X, y):
        X = np.column_stack((X, np.ones(X.shape[0])))
        r, c = X.shape
        
        K = len(np.unique(y))
        np.random.seed(42)
        beta_init = np.random.randn((c) * (K - 1))

        result = fmin_l_bfgs_b(self.neg_log_likelihood, beta_init, args=(X, y), approx_grad=True, maxfun=100000, maxiter=10000)
        result_betas = result[0]
        self.betas = result_betas.reshape(c, (K-1))
        return self

    def predict(self, X):
        X = np.column_stack((X, np.ones(X.shape[0])))
        dot = X@self.betas
        dot = np.column_stack((dot, np.zeros(X.shape[0])))
        exp = np.exp(dot)
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        return probabilities
        
    
    def neg_log_likelihood(self, parameters, *args):
        
        X = args[0]
        y = args[1]

        r, c = X.shape
        K = len(np.unique(y))
        
        beta_array = parameters
        beta_matrix = beta_array.reshape(c, K-1)
        dot = X@beta_matrix
        dot = np.column_stack((dot, np.zeros(r)))

        exp = np.exp(dot)
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        ys = np.eye(K)[y]
        log_likelihood = np.sum(np.log(np.sum(probabilities*ys, axis=1)))
        return -log_likelihood

class OrdinalLogReg():

    def __init__(self):
        self.betas = None
        self.deltas = []

    def build(self, X, y):
        
        r, c = X.shape
        K = len(np.unique(y))

        betas = np.random.randn(c)
        beta_constraints = np.array([(-np.inf, np.inf)] * c)
        
        deltas = np.ones(K-2)
        delta_constraints = np.array([(1, np.inf)]*(K-2))

        parameters = betas if len(deltas) == 0 else np.concatenate((betas, deltas))
        constraints = beta_constraints if len(delta_constraints) == 0 else np.concatenate((beta_constraints, delta_constraints))

        result = fmin_l_bfgs_b(self.neg_log_likelihood, parameters, args=(X, y), approx_grad=True, bounds=constraints)
        self.betas = result[0][:c]
        self.deltas = result[0][c:]
        return self
    
    def predict(self, X):
        
        r, c = X.shape
        K = len(self.deltas)+2

        dots = X@self.betas
        ts = np.zeros(len(self.deltas)+3)
        ts[0] = -np.inf
        ts[1] = 0
        for i, delta in enumerate(self.deltas):
            ts[i+2] = ts[i+1] + delta
        ts[-1] = np.inf

        probabilities = np.zeros((r, K))

        for j, dot in enumerate(dots):
            for i in range(1, len(ts)):
                probabilities[j, i-1] = logistic(ts[i]- dot) - logistic(ts[i-1]- dot)
        return probabilities
    
    def neg_log_likelihood(self, parameters, *args):
        X = args[0]
        y = args[1]

        r, c = X.shape
        K = len(np.unique(y))
        betas = parameters[:c]  
        deltas = parameters[c:]

        dots = X@betas
        ts = np.zeros(len(deltas)+3)
        ts[0] = -np.inf
        ts[1] = 0
        for i, delta in enumerate(deltas):
            ts[i+2] = ts[i+1] + delta
        ts[-1] = np.inf

        log_likelihood = 0

        for j, dot in enumerate(dots):
            log_likelihood += np.log(logistic(ts[y[j]+1]- dot) - logistic(ts[y[j]]- dot))

        return -log_likelihood


def logistic(x):
    return(1/(1+np.exp(-x)))

if __name__ == "__main__":
    pass