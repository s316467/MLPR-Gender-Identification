import numpy as np

def compute_mean_cov(Dtrain, Ltrain, K):
    
    mu_classes = []  # list of empiracal means for each class
    diag_cov_classes = []  # diagonal covariance matrices for each class
    cov_classes = []

    for label in K:
        Dtrain_i = Dtrain[:, Ltrain == label]
        N_i = Dtrain_i.shape[1]
        mu_i = Dtrain_i.mean(axis=1).reshape(-1, 1)
        cov_i = 1 / N_i * np.dot(Dtrain_i - mu_i, (Dtrain_i - mu_i).T)
        mu_classes.append(mu_i)
        cov_classes.append(cov_i)

    for i in range(K):
        diag_cov_classes.append(cov_classes[i] * np.identity(Dtrain.shape[0]))
    
    return mu_classes, diag_cov_classes

def logpdf_GAU_ND_1sample( x, mu, C):
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

    mu_classes, diag_cov_classes = compute_mean_cov(Dtrain, Ltrain, K)
    
    for i in range(Ntest):
        xt = Dtest[:, i:i + 1]
        score = np.zeros(shape=(K, 1))
        for j in range(K):
            mu = mu_classes[j]
            C = diag_cov_classes[j]
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