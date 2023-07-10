import numpy as np
import scipy 

def compute_mean_cov(Dtrain, Ltrain, K):
    
    mu_classes = []
    cov_classes = []

    for label in K:
        Dtrain_i = Dtrain[:, Ltrain == label]
        N_i = Dtrain_i.shape[1]
        mu_i = Dtrain_i.mean(axis=1).reshape(-1, 1)
        cov_i = 1 / N_i * np.dot(Dtrain_i - mu_i, (Dtrain_i - mu_i).T)
        mu_classes.append(mu_i)
        cov_classes.append(cov_i)
    
    return mu_classes, cov_classes

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
    mu_classes, cov_classes = compute_mean_cov(Dtrain, Ltrain, K)

    for i in range(Ntest):
        xt = Dtest[:, i:i + 1]
        score = np.zeros(shape=(K, 1))
        for j in range(K):
            mu = mu_classes[j]
            C = cov_classes[j]
            score[j, :] = np.exp(logpdf_GAU_ND_1sample(xt, mu, C))
        S[:, i:i + 1] = score
   
    SJoint = 1 / 2 * S
    logSJoint = np.log(SJoint) + np.log(1 / 2)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0).reshape(1, -1)
    log_SPost = logSJoint - logSMarginal
    SPost = np.exp(log_SPost)
    predicted_labels = np.argmax(SPost, axis=0)
    return predicted_labels