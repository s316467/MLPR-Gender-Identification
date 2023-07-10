import numpy as np
import scipy 


def compute_mean_cov( Dtrain, Ltrain, K):
    
    cov_classes = []
    mu_classes = []  # list of empiracal means for each class
    tied_cov = 0 # tied covariance matrix
    num_samples_per_class = [sum(Ltrain == i) for i in range(K)]

    for label in K:
        Dtrain_i = Dtrain[:, Ltrain == label]
        N_i = Dtrain_i.shape[1]
        mu_i = Dtrain_i.mean(axis=1).reshape(-1, 1)
        cov_i = 1 / N_i * np.dot(Dtrain_i - mu_i, (Dtrain_i - mu_i).T)
        mu_classes.append(mu_i)
        cov_classes.append(cov_i)
        tied_cov += (num_samples_per_class[label] * cov_classes[label])

    
    tied_cov *= 1 / sum(num_samples_per_class)
    
    return mu_classes, tied_cov

def logpdf_GAU_ND_1sample(x, mu, C):
    M = x.shape[0]
    mu = mu.reshape(M, 1)
    xc = x - mu
    invC = np.linalg.inv(C)
    _, log_abs_detC = np.linalg.slogdet(C)
    return -M / 2 * np.log(2 * np.pi) - 1 / 2 * log_abs_detC - 1 / 2 * np.dot(np.dot(xc.T, invC), xc)

def predict(Dtrain, Ltrain, Dtest):
    Ntest = Dtest.shape[1] # number of test samples
    K = set(Ltrain)
    S = np.zeros(shape=(len(K), Ntest))
    mu_classes, tied_cov = compute_mean_cov(Dtrain, Ltrain, K)

    for i in range(Ntest):
        xt = Dtest[:, i:i + 1]
        score = np.zeros(shape=(K, 1))
        for j in range(K):
            mu = mu_classes[j]
            C = tied_cov
            score[j, :] = np.exp(logpdf_GAU_ND_1sample(xt, mu, C))
        S[:, i:i + 1] = score
    
    SJoint = 1 / 2 * S
    SMarginal = SJoint.sum(0).reshape(-1, 1)
    SPost = np.zeros(shape=(K, Ntest))
    for c in range(K):
        SJoint_c = SJoint[c, :].reshape(-1, 1)
        SPost_c = (SJoint_c / SMarginal).reshape(1, -1)
        SPost[c, :] = SPost_c
    predicted_labels = np.argmax(SPost, axis=0)
    return predicted_labels
    
"""
def get_decision_function_parameters():
    precision_matrix = np.linalg.inv(tied_cov)
    b = np.dot(precision_matrix, (mu_classes[1] - mu_classes[0]))
    c = -1/2 * (np.dot(np.dot(mu_classes[1].T, precision_matrix), mu_classes[1])-
                np.dot(np.dot(mu_classes[0].T, precision_matrix), mu_classes[0]))
    return b, c
"""