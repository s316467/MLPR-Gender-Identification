import numpy
import string
import scipy.linalg
import scipy.special
import matplotlib.pyplot as plt
import sklearn.datasets
import itertools
import sys
import os
 
def mcol(v):
    return v.reshape((v.size, 1))

def vcol(v):
    return numpy.array(v).reshape(-1,1)

def vrow(x):
    return x.reshape(1, -1) # row vector, with the same number of columns as x and 1 row


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def split_data(l, n):

    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])
            
    return lTrain, lTest


def logreg_obj_wrap(DTR, LTR, l):
    dim = DTR.shape[0]
    ZTR = LTR * 2.0 - 1.0
    def logreg_obj(v):
        w = vcol(v[0:dim])
        b = v[-1]
        scores = numpy.dot(w.T, DTR) + b
        loss_per_sample = numpy.logaddexp(0, -ZTR * scores)
        loss = loss_per_sample.mean() + 0.5 * l * numpy.linalg.norm(w)**2
        return loss

def load_iris_binary():
    from sklearn import datasets
    iris = datasets.load_iris()
    D = iris.data
    L = iris.target
    D = D[L != 2, :]
    L = L[L != 2]
    return D.T, L
    
import scipy.optimize
def train_logreg(D, L, lamb):
    logreg_obj = logreg_obj_wrap(D, L, lamb)
    x0 = numpy.zeros(D.shape[0] + 1)
    xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)
    print (xOpt)
    print (fOpt)
    # w, b = vcol(xOpt[0:DTR.shape[0]]), xOpt[-1]
    return xOpt

class logRegModel():
    
    def __init__(self, D, L, l):
        self.DTR = D
        self.ZTR = L * 2.0 - 1.0
        self.l = l
        self.dim = D.shape[0]
    
    def logreg_obj(self, v):
        w = vcol(v[0:self.dim])
        b = v[-1]
        scores = numpy.dot(w.T, self.DTR) + b
        loss_per_sample = numpy.logaddexp(0, -self.ZTR * scores)
        loss = loss_per_sample.mean() + 0.5 * self.l * numpy.linalg.norm(w)**2
        return loss

    def train(self):
        
        x0 = numpy.zeros(self.DTR.shape[0] + 1)
        xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0, approx_grad=True)
        print (xOpt)
        print (fOpt)
        return xOpt
    

### Solution 1 - Dictionaries of frequencies ###

def S1_buildDictionary(lTercets):

    '''
    Create a set of all words contained in the list of tercets lTercets
    lTercets is a list of tercets (list of strings)
    '''

    sDict = set([])
    for s in lTercets:
        words = s.split()
        for w in words:
            sDict.add(w)
    return sDict

def S1_estimateModel(hlTercets, eps = 0.1):

    '''
    Build frequency dictionaries for each class.

    hlTercets: dict whose keys are the classes, and the values are the list of tercets of each class.
    eps: smoothing factor (pseudo-count)

    Return: dictionary h_clsLogProb whose keys are the classes. For each class, h_clsLogProb[cls] is a dictionary whose keys are words and values are the corresponding log-frequencies (model parameters for class cls)
    '''

    # Build the set of all words appearing at least once in each class
    sDictCommon = set([])

    for cls in hlTercets: # Loop over class labels
        lTercets = hlTercets[cls]
        sDictCls = S1_buildDictionary(lTercets)
        sDictCommon = sDictCommon.union(sDictCls)

    # Initialize the counts of words for each class with eps
    h_clsLogProb = {}
    for cls in hlTercets: # Loop over class labels
        h_clsLogProb[cls] = {w: eps for w in sDictCommon} # Create a dictionary for each class that contains all words as keys and the pseudo-count as initial values

    # Estimate counts
    for cls in hlTercets: # Loop over class labels
        lTercets = hlTercets[cls]
        for tercet in lTercets: # Loop over all tercets of the class
            words = tercet.split()
            for w in words: # Loop over words of the given tercet
                h_clsLogProb[cls][w] += 1
            
    # Compute frequencies
    for cls in hlTercets: # Loop over class labels
        nWordsCls = sum(h_clsLogProb[cls].values()) # Get all occurrencies of words in cls and sum them. this is the number of words (including pseudo-counts)
        for w in h_clsLogProb[cls]: # Loop over all words
            h_clsLogProb[cls][w] = numpy.log(h_clsLogProb[cls][w]) - numpy.log(nWordsCls) # Compute log N_{cls,w} / N

    return h_clsLogProb

def S1_compute_logLikelihoods(h_clsLogProb, text):

    '''
    Compute the array of log-likelihoods for each class for the given text
    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    The function returns a dictionary of class-conditional log-likelihoods
    '''
    
    logLikelihoodCls = {cls: 0 for cls in h_clsLogProb}
    for cls in h_clsLogProb: # Loop over classes
        for word in text.split(): # Loop over words
            if word in h_clsLogProb[cls]:
                logLikelihoodCls[cls] += h_clsLogProb[cls][word]
    return logLikelihoodCls

def S1_compute_logLikelihoodMatrix(h_clsLogProb, lTercets, hCls2Idx = None):

    '''
    Compute the matrix of class-conditional log-likelihoods for each class each tercet in lTercets

    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    lTercets is a list of tercets (list of strings)
    hCls2Idx: map between textual labels (keys of h_clsLogProb) and matrix rows. If not provided, automatic mapping based on alphabetical oreder is used
   
    Returns a #cls x #tercets matrix. Each row corresponds to a class.
    '''
    
    if hCls2Idx is None:
        hCls2Idx = {cls:idx for idx, cls in enumerate(sorted(h_clsLogProb))}

    S = numpy.zeros((len(h_clsLogProb), len(lTercets)))
    for tIdx, tercet in enumerate(lTercets):
        hScores = S1_compute_logLikelihoods(h_clsLogProb, tercet)
        for cls in h_clsLogProb: # We sort the class labels so that rows are ordered according to alphabetical order of labels
            clsIdx = hCls2Idx[cls]
            S[clsIdx, tIdx] = hScores[cls]

    return S

### Solution 2 - Arrays of occurrencies ###

def S2_buildDictionary(lTercets):

    '''
    Create a dictionary of all words contained in the list of tercets lTercets
    The dictionary allows storing the words, and mapping each word to an index i (the corresponding index in the array of occurrencies)

    lTercets is a list of tercets (list of strings)
    '''

    hDict = {}
    nWords = 0
    for tercet in lTercets:
        words = tercet.split()
        for w in words:
            if w not in hDict:
                hDict[w] = nWords
                nWords += 1
    return hDict

def S2_estimateModel(hlTercets, eps = 0.1):

    '''
    Build word log-probability vectors for all classes

    hlTercets: dict whose keys are the classes, and the values are the list of tercets of each class.
    eps: smoothing factor (pseudo-count)

    Return: tuple (h_clsLogProb, h_wordDict). h_clsLogProb is a dictionary whose keys are the classes. For each class, h_clsLogProb[cls] is an array containing, in position i, the log-frequency of the word whose index is i. h_wordDict is a dictionary that maps each word to its corresponding index.
    '''

    # Since the dictionary also includes mappings from word to indices it's more practical to build a single dict directly from the complete set of tercets, rather than doing it incrementally as we did in Solution S1
    lTercetsAll = list(itertools.chain(*hlTercets.values())) 
    hWordDict = S2_buildDictionary(lTercetsAll)
    nWords = len(hWordDict) # Total number of words

    h_clsLogProb = {}
    for cls in hlTercets:
        h_clsLogProb[cls] = numpy.zeros(nWords) + eps # In this case we use 1-dimensional vectors for the model parameters. We will reshape them later.
    
    # Estimate counts
    for cls in hlTercets: # Loop over class labels
        lTercets = hlTercets[cls]
        for tercet in lTercets: # Loop over all tercets of the class
            words = tercet.split()
            for w in words: # Loop over words of the given tercet
                wordIdx = hWordDict[w]
                h_clsLogProb[cls][wordIdx] += 1 # h_clsLogProb[cls] ius a 1-D array, h_clsLogProb[cls][wordIdx] is the element in position wordIdx

    # Compute frequencies
    for cls in h_clsLogProb.keys(): # Loop over class labels
        vOccurrencies = h_clsLogProb[cls]
        vFrequencies = vOccurrencies / vOccurrencies.sum()
        vLogProbabilities = numpy.log(vFrequencies)
        h_clsLogProb[cls] = vLogProbabilities

    return h_clsLogProb, hWordDict
    
def S2_tercet2occurrencies(tercet, hWordDict):
    
    '''
    Convert a tercet in a (column) vector of word occurrencies. Word indices are given by hWordDict
    '''
    v = numpy.zeros(len(hWordDict))
    for w in tercet.split():
        if w in hWordDict: # We discard words that are not in the dictionary
            v[hWordDict[w]] += 1
    return mcol(v)

def S2_compute_logLikelihoodMatrix(h_clsLogProb, hWordDict, lTercets, hCls2Idx = None):

    '''
    Compute the matrix of class-conditional log-likelihoods for each class each tercet in lTercets

    h_clsLogProb and hWordDict are the dictionary of model parameters and word indices as returned by S2_estimateModel
    lTercets is a list of tercets (list of strings)
    hCls2Idx: map between textual labels (keys of h_clsLogProb) and matrix rows. If not provided, automatic mapping based on alphabetical oreder is used
   
    Returns a #cls x #tercets matrix. Each row corresponds to a class.
    '''

    if hCls2Idx is None:
        hCls2Idx = {cls:idx for idx, cls in enumerate(sorted(h_clsLogProb))}
    
    numClasses = len(h_clsLogProb)
    numWords = len(hWordDict)

    # We build the matrix of model parameters. Each row contains the model parameters for a class (the row index is given from hCls2Idx)
    MParameters = numpy.zeros((numClasses, numWords)) 
    for cls in h_clsLogProb:
        clsIdx = hCls2Idx[cls]
        MParameters[clsIdx, :] = h_clsLogProb[cls] # MParameters[clsIdx, :] is a 1-dimensional view that corresponds to the row clsIdx, we can assign to the row directly the values of another 1-dimensional array

    SList = []
    for tercet in lTercets:
        v = S2_tercet2occurrencies(tercet, hWordDict)
        STercet = numpy.dot(MParameters, v) # The log-lieklihoods for the tercets can be computed as a matrix-vector product. Each row of the resulting column vector corresponds to M_c v = sum_j v_j log p_c,j
        SList.append(numpy.dot(MParameters, v))

    S = numpy.hstack(SList)
    return S


################################################################################

def compute_classPosteriors(S, logPrior = None):

    '''
    Compute class posterior probabilities

    S: Matrix of class-conditional log-likelihoods
    logPrior: array with class prior probability (shape (#cls, ) or (#cls, 1)). If None, uniform priors will be used

    Returns: matrix of class posterior probabilities
    '''

    if logPrior is None:
        logPrior = numpy.log( numpy.ones(S.shape[0]) / float(S.shape[0]) )
    J = S + mcol(logPrior) # Compute joint probability
    ll = scipy.special.logsumexp(J, axis = 0) # Compute marginal likelihood log f(x)
    P = J - ll # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    return numpy.exp(P)

def compute_accuracy(P, L):

    '''
    Compute accuracy for posterior probabilities P and labels L. L is the integer associated to the correct label (in alphabetical order)
    '''

    PredictedLabel = numpy.argmax(P, axis=0)
    NCorrect = (PredictedLabel.ravel() == L.ravel()).sum()
    NTotal = L.size
    return float(NCorrect)/float(NTotal)


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def mean_cov_estimate(X):
    mu = vcol(X.mean(1))
    C = numpy.dot(X-mu, (X-mu).T)/X.shape[1]
    return mu, C

def logpdf_GAU_ND_1Sample(X, mu, C):
    xc = X - mu
    M = X.shape[0]
    const = -0.5 * M * numpy.log(2 * numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]
    L = numpy.linalg.inv(C)
    v = numpy.dot(xc.T, numpy.dot(L, xc)).ravel()
    return const - 0.5 * logdet - 0.5 * v

def logpdf_GAU_ND(X, mu, C): # from lab4
    Y = []
    for i in range (X.shape[1]):
        Y.append(logpdf_GAU_ND_1Sample(X[:,i:i+1], mu, C))
    return numpy.array(Y).ravel()

def logpdf_GAU_ND(X, mu, C): # from lab5
    # X is a M x N matrix, mu is a M x 1 vector, C is a M x M matrix
    M, N = X.shape # M is the dimensionality of the data, that is the number of rows of X, that is 4, N is the number of samples, that is the number of columns of X, that is 4
    assert mu.shape == (M, 1), "mu must have shape (M, 1)" # mu is a M x 1 vector
    assert C.shape == (M, M), "C must have shape (M, M)" # C is a M x M matrix

    # Compute the inverse and log-determinant of covariance matrix C
    C_inv = numpy.linalg.inv(C) # C_inv is a M x M matrix (inverse of C)
    _, logdet_C = numpy.linalg.slogdet(C)  # logdet_C is a scalar (log-determinant of C), the second value returned by numpy.linalg.slogdet is the absolute value of the log-determinant
                                        # while the first one is the sign of the determinant, which, for covariance matrices, is positive

    # Initialize the log-densities array
    logpdfs = numpy.zeros(N) # logpdfs is a 1 x N vector, where N is the number of samples, and we are initializing it with zeros
    X_diff = X - mu  # X_diff is an M x N matrix, where each column is the difference between the corresponding column of X and mu
    term1 = -0.5 * numpy.sum(numpy.multiply(numpy.dot(C_inv, X_diff), X_diff), axis=0)  # term1 is a 1 x N vector, the result of the element-wise multiplication of C_inv*X_diff and X_diff, summed along axis 0
    logpdfs = term1 - 0.5 * (M * numpy.log(2 * numpy.pi) + logdet_C)  # logpdfs is a 1 x N vector, where N is the number of samples

    return logpdfs


def logpdf_GAU_ND_fast(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = -0.5 * M * numpy.log(2 * numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]
    L = numpy.linalg.inv(C)
    v = (XC*numpy.dot(L, XC)).sum(0)
    return const - 0.5 * logdet - 0.5 * v

def loglikelihood(X, mu, C): # X is a M x N matrix, mu is a M x 1 vector, C is a M x M matrix
    return numpy.sum(logpdf_GAU_ND(X, mu, C)) # log-likelihood is the sum of the log-densities

def classify_iris():
    D, L = load_iris() # to be modified
    (DTR, LTR), (DTV, LTV) = split_db_2to1(D, L)
    hCls = {}
    for lab in [0,1,2]:
        DCLS = DTR[:, LTR==lab]
        hCls[lab] = mean_cov_estimate(DCLS)
    
    ## Classification
    prior = vcol(numpy.ones(3)/3.0)
    S = []
    for hyp in [0,1,2]:
        mu, C = hCls[hyp]
        fcond = numpy.exp(logpdf_GAU_ND(DTV, mu, C))
        S.append(vrow(fcond))
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    
    logSJoint = numpy.zeros((3, DTV.shape[1]))
    for hyp in [0,1,2]:
        logSJoint[hyp, :] = logpdf_GAU_ND(DTV, hCls[hyp][0], hCls[hyp][1]) + numpy.log(prior[hyp])
    
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    predicted_labels = numpy.argmax(P, axis=0)
    accuracy = numpy.sum(predicted_labels == LTV) / LTV.shape[0]
    print( "Accuracy: ", accuracy)
    err = 1 - accuracy
    print( "Error rate: ", err)
    return S, P


def classify_log_iris():
    D, L = load_iris()
    (DTR, LTR), (DTV, LTV) = split_db_2to1(D, L)
    hCls =  {}
    for lab in [0,1,2]:
       DCLS = DTR[:, LTR==lab]
       hCls[lab] = mean_cov_estimate(DCLS)
    
    ### Classification
    logprior = numpy.log(vcol(numpy.ones(3)/3.0))
    S = []
    for hyp in [0,1,2]:
        mu, C = hCls[hyp]
        fcond = logpdf_GAU_ND(DTV, mu, C)
        S.append(vrow(fcond))
    S = numpy.vstack(S)
    S = S + logprior
    logP = S - vrow(scipy.special.logsumexp(S, 0))
    P = numpy.exp(logP)
    
    # here from sol screens you should return S and P
    
    logSJoint = numpy.zeros((3, DTV.shape[1]))
    for hyp in [0,1,2]:
        logSJoint[hyp, :] = logpdf_GAU_ND(DTV, hCls[hyp][0], hCls[hyp][1]) + logprior[hyp]
    
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    
    predicted_labels = numpy.argmax(P, axis=0)
    accuracy = numpy.sum(predicted_labels == LTV) / LTV.shape[0]
    print( "Accuracy: ", accuracy)
    err = 1 - accuracy
    print( "Error rate: ", err)
    return S, P

def PCA(D, m):
    mu = vcol(D.mean(1))
    C = numpy.dot(D-mu, (D-mu).T)/D.shape[1]
    s,U = numpy.linalg.eigh(C)
    U = U[:, ::-1]
    P = U[:, 0:m]
    
    return P

def PCA2(D, m):
    mu = vcol(D.mean(1))
    C = numpy.dot(D-mu, (D-mu).T)/D.shape[1]
    U, _, _ = numpy.linalg.svd(C)
    P = U[:, 0:m]
    
    return P

def SbSw(D, L):
    SB = 0
    SW = 0
    mu = vcol(D.mean(1))
    for i in range(L.max()+1):
        DCls = D[:, L == i]
        muCls = vcol(DCls.mean(1))
        SW += numpy.dot(DCls - muCls, (DCls - muCls).T)
        SB += DCls.shape[1] * numpy.dot(muCls - mu, (muCls - mu).T)
    SW /= D.shape[1]
    SB /= D.shape[1]
    
    return SB, SW

def LDA1(D, L, m):
    SB, SW = SbSw(D, L)
    s, U = scipy.linalg.eigh(SB, SW)
    
    return U[:, ::-1][:, 0:m]

def LDA2(D,L,m):
    SB, SW = SbSw(D,L)
    U, s, _ = numpy.linalg.svd(SW)
    P1 = numpy.dot(U, vcol(1.0/s**0.5)*U.T)
    SBTilde = numpy.dot(P1, numpy.dot(SB, P1.T))
    U, _, _ = numpy.linalg.svd(SBTilde)
    P2 = U[:, 0:m]
    
    return numpy.dot(P1.T, P2)
    
def plot_reduced_data(data, labels, method_name, ax):
    classes = numpy.unique(labels)
    colors = ['r', 'g', 'b']

    for cls, color in zip(classes, colors):
        ax.scatter(data[0, labels == cls], data[1, labels == cls], c=color, label=f'Class {cls}')
    
    ax.set_title(f'{method_name} Reduced Data')
    ax.legend()

def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for dIdx in range(4):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Setosa')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Versicolor')
        plt.hist(D2[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Virginica')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()
