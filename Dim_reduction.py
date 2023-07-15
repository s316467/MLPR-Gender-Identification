import numpy as np
import ReadData
import matplotlib.pyplot as plt

def computeProjectionMatrix(D, m):
    # center the dataset in the origin
    mu = ReadData.vcol(D.mean(1))
    Dc = D - mu
    # compute the covariance matrix
    C = np.dot(Dc, Dc.T)/D.shape[1]
    # get eigenvectors
    _ , U = np.linalg.eigh(C)
    # keep only the first m eigenvectors
    P = U[:, ::-1][:, 0:m]  
    return P

def PCA (D , P):

    DP = np.dot(P.T, D)
    # return projected dataset
    return DP
    
def computeSigmaMatrix(D):
    # center the dataset in the origin
    mu = ReadData.vcol(D.mean(1))
    DC = D - mu
    # compute the matrix
    C = np.dot(DC,DC.T)/D.shape[1]
    sigma, _ = np.linalg.eigh(C)
    return sigma

def scatter_2D_plot(DT_pca, LT):
    plt.figure()
    plt.scatter(DT_pca[0, LT == 0], DT_pca[1, LT == 0])
    plt.scatter(DT_pca[0, LT == 1], DT_pca[1, LT == 1])
    plt.show()

def kfold_PCA(D=None, k=3, threshold=0.95, show=False):

    #preparing folds
    np.random.seed(0)
    idx = np.random.permutation( D.shape[1] )
    folds = np.array_split(idx, k)

    #num of m direction
    m_values = np.arange(1, D.shape[0])[::-1]
    avg_perc_values = []

    for m in m_values:

        avg_perc = 0
        for i in range(k):
            fold_test = folds[i]
            folds_train = []
            for j in range(k):
                if j != i:
                    folds_train.append(folds[j])

            Dtrain = D[:, np.array(folds_train).flat]
            
            #obtain the eigenvalues from sigma and compute the percentage of the variance of data given a certain m
            sigma = computeSigmaMatrix(Dtrain)
            largest_eigh = np.flip(sigma)[0:m]
            t = sum(largest_eigh) / sum(sigma)
            avg_perc += t
        
        #this will compute an array of avg for the different values of m that represent the variance of the data
        avg_perc /= k
        avg_perc_values.append(avg_perc)

    if show:
        plt.figure()
        plt.plot(m_values, np.array(avg_perc_values) * 100, '--o')
        plt.xticks(m_values)
        plt.xlabel('m')
        plt.ylabel('% of retained variance of data')
        plt.show()
    return m_values, np.array(avg_perc_values)
