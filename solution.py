import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd

X = np.arange(0, 10, 1)
M = 2

B = 3

Y = M * X + B
initial_values = np.array([0.0, 1.0])
def fun(parameters, *args):
        x = args[0]
        y = args[1]
        m, b = parameters
        y_model = m*x + b
        error = sum(np.power((y - y_model), 2))
        return error





class MultinomialLogReg():

    def __init__(self):
        self.encoder = {}
        self.decoder = {}
        self.betas = None

    def build(self, X, y):
        
        r, c = X.shape
        self.word_to_index(y)
        K = len(self.encoder)
        beta_init = np.random.randn(c * (K - 1))

        result = fmin_l_bfgs_b(self.maximum_likelihood, beta_init, args=(X, y), approx_grad=True)
        result_betas = result[0]
        self.betas = result_betas.reshape(c, (K-1))
        return self

    def predict(self, X):
        
        dot = X@self.betas
        dot = np.column_stack((dot, np.zeros(X.shape[0])))
        exp = np.exp(dot)
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        return probabilities
        
    
    def maximum_likelihood(self, parameters, *args):
        
        X = args[0]
        y = args[1]

        r, c = X.shape
        y_encoded = self.one_hot_encode(y)
        K = len(self.encoder)
        
        beta_array = parameters
        beta_matrix = beta_array.reshape(c, K-1)
        dot = X@beta_matrix
        dot = np.column_stack((dot, np.zeros(r)))
    
        exp = np.exp(dot)
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        log_likelihood = np.sum(np.log(np.sum(y_encoded*probabilities, axis=1)))
        return - log_likelihood

    def one_hot_encode(self, y):
        y_encoded = np.zeros((len(y), len(self.encoder)))
        for i, word in enumerate(y):
            y_encoded[i][self.encoder[word]] = 1
        return y_encoded
            
    
    def word_to_index(self, y):
        unique = np.unique(y)
        for index, label in enumerate(unique):
            self.encoder[label] = index
            self.decoder[index] = label

if __name__ == "__main__":
    df = pd.read_csv("dataset.csv", delimiter=";")
    y = df["ShotType"].to_numpy()
    X = df.drop(columns="ShotType").to_numpy()
    rg = MultinomialLogReg()
    rg.build(X, y)



    