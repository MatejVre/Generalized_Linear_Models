import random
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.special import erfinv
import seaborn as sns



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
        
        X = np.column_stack((X, np.ones(X.shape[0])))
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
        
        X = np.column_stack((X, np.ones(X.shape[0])))
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



class multinomial_for_scikit(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.model = MultinomialLogReg()
        self.model.build(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)
    
    def predict_proba(self, X):
        return self.model.predict(X)
    
class ordinal_for_scikit(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.model = OrdinalLogReg()
        self.model.build(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)
    
    def predict_proba(self, X):
        return self.model.predict(X)

class LinearRegression():

    def __init__(self):
        self.betas = None
        self.hat_matrix = None
    
    def build(self, X, y):
        X = np.column_stack((X, np.ones(X.shape[0])))
        matr_for_both = np.linalg.inv((X.T@X))@X.T
        self.betas = matr_for_both@y
        self.hat_matrix = X @ matr_for_both

    def predict(self, X):
        X = np.column_stack((X, np.ones(X.shape[0])))
        return X@self.betas


transformer = ColumnTransformer(
    transformers=[
        ("pass", "passthrough", ["Transition"]),
        ("cat1", OneHotEncoder(drop=["EURO"]), ["Competition"]),
        ("cat2", OneHotEncoder(drop=["dribble or cut"]), ["Movement"]),
        ("cat3", OneHotEncoder(drop=["G"]), ["PlayerType"]),
        ("scaled", StandardScaler(), ["Angle", "Distance"])
    ]
)

def perform_lr(X, y):
    lr = LinearRegression()
    lr.build(X, y)
    b = lr.betas
    plt.scatter(X, y, marker="o", facecolor="none", edgecolors="black", s=13)
    space = np.linspace(np.min(X), np.max(X))
    Y = b[0]*space + b[1]
    plt.plot(space, Y, color="r")
    plt.xlabel("Angle")
    plt.ylabel("Distance")
    plt.savefig("linear_regression.png", bbox_inches="tight")
    plt.show()

def q_q_plot(lr, X, y):
    predictions = lr.predict(X)
    residuals = y - predictions
    s = np.std(residuals)
    quantile_residuals = residuals/s
    sorted_residuals = np.sort(quantile_residuals)
    n = len(sorted_residuals)
    probabilities = (np.arange(0, n)) / n
    theoretical_quantiles = np.sqrt(2) * erfinv(2 * probabilities - 1)
    plt.figure(figsize=(6, 6))
    plt.scatter(theoretical_quantiles, sorted_residuals, edgecolor='black', facecolor="none", marker="o", s=40)
    plt.plot(theoretical_quantiles, theoretical_quantiles, color='red', linestyle="--")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.savefig("qqplot", bbox_inches="tight")
    plt.show()

def fitted_vs_residuals(lr, X, y):
    predictions = lr.predict(X)
    residuals = y - predictions
    n = len(residuals)
    sigma_hat = np.sqrt(np.sum(residuals**2) / (n - 2 - 1))
    std = np.std(residuals)
    print(sigma_hat, std)
    h = np.diag(lr.hat_matrix)
    standardized_residuals = residuals / (std*np.sqrt(1-h))
    sns.regplot(x=predictions, y=standardized_residuals, lowess=True, scatter_kws={'s': 20, 'color':"black", "facecolor":"None"}, line_kws={'color': 'red'})
    plt.axhline(0, c="blue", ls="--")
    plt.xlabel("Fitted Values")
    plt.ylabel("Standardized Residuals")
    lowess_line = Line2D([0], [0], color='red', lw=2, label='LOWESS')
    plt.legend(handles=[lowess_line])
    plt.savefig("homo.png", bbox_inches="tight")
    plt.show()

def cooks_distance(lr, X, y):
    predictions = lr.predict(X)
    residuals = y - predictions
    k = len(predictions)
    MSE = np.sum(residuals**2)/k
    h = np.diag(lr.hat_matrix)
    D = (residuals**2 / (2*(MSE))) * (h / ((1 - h))**2)
    print(D[0:15])
    plt.bar(range(k), D)
    plt.xlabel("Point")
    plt.ylabel("Cook's Distance")
    plt.savefig("cooks.png", bbox_inches="tight")
    plt.show()

def bootstrap_pipeline(df):
    n_rows = df.shape[0]
    indices = range(n_rows)
    bootstrap_repetitions = 100
    all_betas = np.zeros((5, 12, bootstrap_repetitions))

    for i in range(bootstrap_repetitions):
        bootstrap_indices = random.choices(indices, k=n_rows)
        bootstrap_dataset = df.loc[bootstrap_indices]

        y = bootstrap_dataset["ShotType"]
        X = bootstrap_dataset.drop(columns="ShotType")

        transformer.fit(X)
        transformed_X = transformer.transform(X)

        encoder = LabelEncoder()
        encoder.fit(y)

        encoded_y = encoder.transform(y)

        mult_reg = MultinomialLogReg()
        mult_reg.build(transformed_X, encoded_y)

        betas = mult_reg.betas.T

        all_betas[:, :, i] = betas

    transformed_df = pd.DataFrame(transformed_X, columns=transformer.get_feature_names_out())
    np.save("all_betas", all_betas)
    return transformer, encoder, all_betas #needed so we can extract the correct column and index names


def clean_feature_labels(labels):
    cleaned = []
    for label in labels:
        if '__' in label:
            cleaned.append(label.split('__')[-1])  # Keep part after last '__'
        else:
            cleaned.append(label)
    return cleaned

def beta_matrix(columns, coefs_df):
    all_betas = np.load("all_betas.npy")
    coefs_array = np.mean(all_betas, axis=2)
    errors_array = np.std(all_betas, axis=2)
    coefs_str = np.char.mod('%.1f', coefs_array.astype(np.float32))  # Format coefficients
    errors_str = np.char.mod('%.1f', errors_array.astype(np.float32))  # Format errors

    # Combine coefficients and errors with " ± " between them
    annot_array = np.core.defchararray.add(coefs_str, '\n+/-')
    annot_array = np.core.defchararray.add(annot_array, errors_str)
    cleaned_columns = clean_feature_labels(columns)
    # Now plot the heatmap with annotations
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(coefs_df, annot=annot_array, center=0, cmap="vlag", fmt="", cbar=True, annot_kws={"fontsize":12})
    ax.set_xticklabels(cleaned_columns, rotation=60, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    plt.savefig("matrix.pdf", bbox_inches="tight")

def multinomial_bad_ordinal_good(n_rows):
    np.random.seed(17)
    hitter_power = np.random.normal(0, 1, n_rows)
    hitter_speed = np.random.normal(0, 1, n_rows)
    noise = np.random.normal(0, 1, n_rows)

    predictor = 2 * hitter_power + 1 * hitter_speed + noise

    thresholds = [-2, 1, 2.5, 4.2]
    y = np.digitize(predictor, thresholds)
    return pd.DataFrame({
        "power":hitter_power,
        "speed":hitter_speed,
        "score":y
    })

def logistic(x):
    return(1/(1+np.exp(-x)))

if __name__ == "__main__":
    # df = pd.read_csv("dataset.csv", delimiter=";")
    # df = df.drop(columns="TwoLegged")
    # y = df["ShotType"]
    # y = LabelEncoder().fit_transform(y)
    # X = df.drop(columns="ShotType")
    
    # transformer.fit(X)
    # transformed_X = transformer.transform(X)
    # columns = transformer.get_feature_names_out()
    # columns = np.append(columns, np.array("Intercept"))

    #BOOTSTRAP AND GET DATA
    # tf, enc, all_betas = bootstrap_pipeline(df)
    # columns = tf.get_feature_names_out()
    # coefs_df = pd.DataFrame(np.mean(all_betas, axis=2), columns=columns, index=enc.inverse_transform(np.arange(all_betas.shape[0])))

    # beta_matrix(columns, coefs_df)

    #CV TO TEST THE MODESL
    # pipeline = Pipeline([
    #     ("preprocessing", transformer),
    #     ("classifier", ordinal_for_scikit())
    # ])

    # ordinal_scores = cross_validate(pipeline, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), scoring=["accuracy", "neg_log_loss"])
    # accs = ordinal_scores["test_accuracy"]
    # mean_accs = np.mean(accs)
    # sd_accs = np.std(accs)
    # se_accs = sd_accs/(np.sqrt(len(accs)))
    # print(f"Ordinal Logistic Regression Accuracy: {mean_accs} +/- {se_accs}")
    # logs = np.abs(ordinal_scores["test_neg_log_loss"])
    # mean_logs = np.mean(logs)
    # sd_logs = np.std(logs)
    # se_logs = sd_logs/(np.sqrt(len(logs)))
    # print(f"Ordinal Logistic Regression Accuracy: {mean_logs} +/- {se_logs}")

    # pipeline = Pipeline([
    # ("preprocessing", transformer),
    # ("classifier", multinomial_for_scikit())
    # ])

    # multi_scores = cross_validate(pipeline, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), scoring=["accuracy", "neg_log_loss"])
    # accs = multi_scores["test_accuracy"]
    # mean_accs = np.mean(accs)
    # sd_accs = np.std(accs)
    # se_accs = sd_accs/(np.sqrt(len(accs)))
    # print(f"Multinomial Logistic Regression Accuracy: {mean_accs} +/- {se_accs}")
    # logs = np.abs(multi_scores["test_neg_log_loss"])
    # mean_logs = np.mean(logs)
    # sd_logs = np.std(logs)
    # se_logs = sd_logs/(np.sqrt(len(logs)))
    # print(f"Multinomial Logistic Regression Accuracy: {mean_logs} +/- {se_logs}")

    #DATA GENERATING PROCESS
    # dgp = multinomial_bad_ordinal_good(300)
    # y = dgp["score"].to_numpy()
    # x = dgp.drop(columns="score").to_numpy()

    # ordinal = ordinal_for_scikit()
    # ordinal_score = cross_validate(ordinal, x, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), scoring="neg_log_loss")
    # print(np.mean(ordinal_score["test_score"]))
    # np.std(ordinal_score["test_score"])/np.sqrt(len(ordinal_score["test_score"]))

    # multinomial = multinomial_for_scikit()
    # multinomial_score = cross_validate(multinomial, x, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), scoring="neg_log_loss")
    # print(np.mean(multinomial_score["test_score"]))
    # np.std(multinomial_score["test_score"])/np.sqrt(len(multinomial_score["test_score"]))

    #PART 3
    # df = pd.read_csv("dataset.csv", delimiter=";")
    # X = df["Angle"]
    # y = df["Distance"]
    # lr = LinearRegression()
    # lr.build(X, y)

    # fitted_vs_residuals(lr, X, y)
    # cooks_distance(lr, X, y)
    # q_q_plot(lr, X, y)
    pass
