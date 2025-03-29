import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class MultinomialLogReg():

    def __init__(self):
        # self.encoder = {}
        # self.decoder = {}
        self.betas = None

    def build(self, X, y):

        r, c = X.shape
        
        K = len(np.unique(y))
        np.random.seed(42)
        beta_init = np.random.randn((c + 1)* (K - 1)) #plus 1 for the intercept

        result = fmin_l_bfgs_b(self.maximum_likelihood, beta_init, args=(X, y), approx_grad=True)
        result_betas = result[0]
        self.betas = result_betas.reshape(c + 1, (K-1))
        print(self.betas)
        return self

    def predict(self, X):
        
        X = np.column_stack((X, np.ones(X.shape[0])))
        dot = X@self.betas
        dot = np.column_stack((dot, np.zeros(X.shape[0])))
        exp = np.exp(dot)
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        return probabilities
        
    
    def maximum_likelihood(self, parameters, *args):
        
        X = args[0]
        y = args[1]

        r, c = X.shape
        X = np.column_stack((X, np.ones(r)))
        # y_encoded = self.one_hot_encode(y) #plus 1 for the intercept
        K = len(np.unique(y))
        
        beta_array = parameters
        beta_matrix = beta_array.reshape(c+1, K-1)
        dot = X@beta_matrix
        dot = np.column_stack((dot, np.zeros(r)))
    
        exp = np.exp(dot)
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        log_likelihood = 0
        for i in range(r):
            log_likelihood += np.log(probabilities[i, y[i]])
        return - log_likelihood

    # def one_hot_encode(self, y):
    #     y_encoded = np.zeros((len(y), len(self.encoder)))
    #     for i, word in enumerate(y):
    #         y_encoded[i][self.encoder[word]] = 1
    #     return y_encoded
            
    
    # def word_to_index(self, y):
    #     unique = np.unique(y)
    #     for index, label in enumerate(unique):
    #         self.encoder[label] = index
    #         self.decoder[index] = label

class OrdinalLogReg():

    def __init__(self):
        self.encoder = {}
        self.decoder = {}
        self.betas = None
        self.deltas = []

    def build(self, X, y):
        
        r, c = X.shape
        self.word_to_index(y)
        K = len(self.encoder)

        betas = np.random.randn(c)
        beta_constraints = np.array([(-np.inf, np.inf)] * c)
        
        deltas = np.ones(K-2)
        delta_constraints = np.array([(0, np.inf)]*(K-2))
        print(delta_constraints)

        parameters = betas if len(deltas) == 0 else np.concatenate((betas, deltas))
        constraints = beta_constraints if len(delta_constraints) == 0 else np.concatenate((beta_constraints, delta_constraints))

        result = fmin_l_bfgs_b(self.maximum_likelihood, parameters, args=(X, y), approx_grad=True, bounds=constraints)
        print(result[0])
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
        print(ts)
        for j, dot in enumerate(dots):
            for i in range(1, len(ts)):
                probabilities[j, i-1] = logistic(ts[i]- dot) - logistic(ts[i-1]- dot)
        return probabilities
    
    def maximum_likelihood(self, parameters, *args):
        X = args[0]
        y = args[1]

        r, c = X.shape
        y_encoded = self.one_hot_encode(y)
        K = len(self.encoder)
        betas = parameters[:c]  
        deltas = parameters[c:]

        dots = X@betas
        #create the thresholds (can it be optimized?)
        ts = np.zeros(len(deltas)+3)
        ts[0] = -np.inf
        ts[1] = 0
        for i, delta in enumerate(deltas):
            ts[i+2] = ts[i+1] + delta
        ts[-1] = np.inf

        probabilities = np.zeros((r, K))

        for j, dot in enumerate(dots):
            for i in range(1, len(ts)):
                probabilities[j, i-1] = logistic(ts[i]- dot) - logistic(ts[i-1]- dot)
        log_likelihood = np.sum(np.log(np.sum(y_encoded*probabilities, axis=1)))
        return -log_likelihood


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

def logistic(x):
    return(1/(1+np.exp(- x)))

if __name__ == "__main__":
    # df = pd.read_csv("dataset.csv", delimiter=";")
    # train, test = train_test_split(df, test_size=0.2)

    # categorical_features = ["Competition", "PlayerType", "Movement"]
    # for feat in categorical_features:
    #     le = LabelEncoder()
    #     le.fit(train[feat])
    #     train[feat] = le.transform(train[feat])
    #     test[feat] = le.transform(test[feat])
    # y_train = train["ShotType"]
    # X_train = train.drop(columns=["ShotType"])

    # y_test = test["ShotType"]
    # X_test = test.drop(columns=["ShotType"])

    # log_reg = OrdinalLogReg()
    # lr = log_reg.build(X_train, y_train)

    # probs = lr.predict(X_test)
    # classes = np.argmax(probs, axis=1)

    # predictions = [lr.decoder[label] for label in classes]
    # accuracy = np.mean(predictions == y_test)
    # print(accuracy)
    pass

    