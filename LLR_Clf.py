import numpy as np
import scipy.special
import scipy.optimize

class LinearLogisticRegression:

    def __init__(self, lbd, prior_weighted=False, prior=0.5):
        self.lbd = lbd
        self.prior_weighted = prior_weighted
        self.prior = prior

    def __compute_zi(self, ci):
        return 2 * ci - 1

    def __logreg_obj(self, v):
        w, b = v[0:-1], v[-1]
        z = 2 * self.Ltrain - 1
        exp = (np.dot(w.T, self.Dtrain) + b)
        reg = 0.5 * self.lbd * np.linalg.norm(w) ** 2
        avg_risk = (np.logaddexp(0, -exp * z)).mean()
        return reg + avg_risk

    def __logreg_obj_prior_weighted(self, v):
        w, b = v[0:-1], v[-1]
        z = 2 * self.Ltrain - 1
        reg = 0.5 * self.lbd * np.linalg.norm(w) ** 2
        exp = (np.dot(w.T, self.Dtrain) + b)
        avg_risk_0 = np.logaddexp(0, -exp[self.Ltrain == 0] * z[self.Ltrain == 0]).mean()*(1-self.prior)
        avg_risk_1 = np.logaddexp(0, -exp[self.Ltrain == 1] * z[self.Ltrain == 1]).mean() * self.prior
        return reg + avg_risk_0 + avg_risk_1

    def train(self, Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.F = Dtrain.shape[0]  # dimensionality of features space
        self.K = len(set(Ltrain))  # number of classes
        self.N = Dtrain.shape[1]
        obj_function = self.__logreg_obj if self.prior_weighted is False else self.__logreg_obj_prior_weighted
        self.x, f, d = scipy.optimize.fmin_l_bfgs_b(func=obj_function,
                                                    x0=np.zeros(self.Dtrain.shape[0] + 1),
                                                    approx_grad=True,
                                                    iprint=0)

        return self

    def get_model_parameters(self):
        w, b = self.x[0:-1], self.x[-1]
        return w, b

    def predict(self, Dtest, labels=True):
        w, b = self.x[0:-1], self.x[-1]
        S = np.zeros((Dtest.shape[1]))
        for i in range(Dtest.shape[1]):
            xi = Dtest[:, i:i + 1]
            s = np.dot(w.T, xi) + b
            S[i] = s
        if labels:
            LP = S > 0
            return LP
        else:
            return S