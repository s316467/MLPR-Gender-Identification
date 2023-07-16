import numpy as np
import scipy
import matplotlib.pyplot as plt

class MVG:
    
    def __init__(self):
        self.mu_classes = []
        self.cov_classes = []

    def train(self, Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.N = Dtrain.shape[1]
        self.F = Dtrain.shape[0]
        self.K = len(set(Ltrain))
        self.labels = set(self.Ltrain)
        self.mu_classes = []
        self.cov_classes = []
        for i in self.labels:
            Dtrain_i = self.Dtrain[:, self.Ltrain == i]
            N_i = Dtrain_i.shape[1]
            mu_i = Dtrain_i.mean(axis=1).reshape(-1, 1)
            cov_i = 1 / N_i * np.dot(Dtrain_i - mu_i, (Dtrain_i - mu_i).T)
            self.mu_classes.append(mu_i)
            self.cov_classes.append(cov_i)
        return self

    def __logpdf_GAU_ND_1sample(self, x, mu, C):
        M = x.shape[0]
        mu = mu.reshape(M, 1)
        xc = x - mu
        invC = np.linalg.inv(C)
        _, log_abs_detC = np.linalg.slogdet(C)
        return -M / 2 * np.log(2 * np.pi) - 1 / 2 * log_abs_detC - 1 / 2 * np.dot(np.dot(xc.T, invC), xc)

    def predict(self, Dtest, labels=True):
        Ntest = Dtest.shape[1]
        S = np.zeros(shape=(self.K, Ntest))
        for i in range(Ntest):
            xt = Dtest[:, i:i + 1]
            score = np.zeros(shape=(self.K, 1))
            for j in range(self.K):
                mu = self.mu_classes[j]
                C = self.cov_classes[j]
                score[j, :] = np.exp(self.__logpdf_GAU_ND_1sample(xt, mu, C))
            S[:, i:i + 1] = score
        if labels:
            SJoint = 1 / 2 * S
            logSJoint = np.log(SJoint) + np.log(1 / 2)
            logSMarginal = scipy.special.logsumexp(logSJoint, axis=0).reshape(1, -1)
            log_SPost = logSJoint - logSMarginal
            SPost = np.exp(log_SPost)
            predicted_labels = np.argmax(SPost, axis=0)
            return predicted_labels
        else:
            return np.log(S[1, :]) - np.log(S[0, :])