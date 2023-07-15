import numpy as np
from scipy.stats import norm
import seaborn
import matplotlib.pyplot as plt

def gaussianized_features_training(DT):
        F = DT.shape[0]
        N = DT.shape[1]
        ranks = []
        for j in range(F):
            rank_num = 0
            for i in range(N):
                rank_num += (DT[j, :] > DT[j, i]).astype(int)
            rank_num += 1
            ranks.append(rank_num / (N+2))
        y = norm.ppf(ranks)
        return y

def gaussianized_features_evaluation(DE, DT):
        F = DT.shape[0]
        NT = DT.shape[1]
        NE = DE.shape[1]
        ranks = []
        for j in range(F):
            rank_num = 0
            for i in range(NT):
                rank_num += (DE[j, :] > DT[j, i]).astype(int)
            rank_num += 1
            ranks.append(rank_num / (NT + 2))
        y = norm.ppf(ranks)
        return y

def znormalized_features_training(DT):
        DTmean = DT.mean(axis=1).reshape(-1, 1)
        DTstdDev = DT.std(axis=1).reshape(-1, 1)
        ZnormDT = (DT - DTmean) / DTstdDev
        return ZnormDT

def znormalized_features_evaluation(DE, DT):
        DTmean = DT.mean(axis=1).reshape(-1, 1)
        DTstdDev = DT.std(axis=1).reshape(-1, 1)
        ZnormDE = (DE - DTmean) / DTstdDev
        return ZnormDE

def heatmap(DT, LT, plt, title):
        fig, axs = plt.subplots(1, 3)
        #fig.suptitle(title)
        seaborn.heatmap(np.corrcoef(DT), linewidth=0.2, cmap="Greys", square=True, cbar=False, ax=axs[0])
        seaborn.heatmap(np.corrcoef(DT[:, LT == 0]), linewidth=0.2, cmap="Blues", square=True, cbar=False, ax=axs[1])
        seaborn.heatmap(np.corrcoef(DT[:, LT == 1]), linewidth=0.2, cmap="Oranges", square=True, cbar=False, ax=axs[2])


def plot_features_hist(DT, LT, preproc='raw', title=True):
        fig1, axs1 = plt.subplots(3, 4)
        k = 0
        if preproc == 'raw':
            if title:
                fig1.suptitle('No preprocessing')
            for i in range(3):
                for j in range(4):
                    axs1[i, j].hist(DT[k, LT == 0], bins=20, alpha=0.5, ec='black', density=True)  # male
                    axs1[i, j].hist(DT[k, LT == 1], bins=20, alpha=0.5, ec='black', density=True)  # female
                    axs1[i, j].set_title('Feature %d' % k)
                    k += 1
        elif preproc == 'gau':
            DTgaussianized = gaussianized_features_training(DT)
            if title:
                fig1.suptitle('Gaussianization preprocessing')
            for i in range(3):
                for j in range(4):
                    axs1[i, j].hist(DTgaussianized[k, LT == 0], bins=20, alpha=0.5, ec='black', density=True)  # male
                    axs1[i, j].hist(DTgaussianized[k, LT == 1], bins=20, alpha=0.5, ec='black', density=True)  # female
                    axs1[i, j].set_title('Feature %d' % k)
                    k += 1
        elif preproc == 'znorm':
            DTznorm = znormalized_features_training(DT)
            if title:
                fig1.suptitle('Z-normalization preprocessing')
            for i in range(3):
                for j in range(4):
                    axs1[i, j].hist(DTznorm[k, LT == 0], bins=20, alpha=0.5, ec='black', density=True)  # male
                    axs1[i, j].hist(DTznorm[k, LT == 1], bins=20, alpha=0.5, ec='black', density=True)  # female
                    axs1[i, j].set_title('Feature %d' % k)
                    k += 1
        elif preproc == 'zg':
            DTznorm = znormalized_features_training(DT)
            DTznormgau = gaussianized_features_training(DTznorm)
            if title:
                fig1.suptitle('Z-normalization + Gaussianization preprocessing')
            for i in range(3):
                for j in range(4):
                    axs1[i, j].hist(DTznormgau[k, LT == 0], bins=20, alpha=0.5, ec='black', density=True)  # male
                    axs1[i, j].hist(DTznormgau[k, LT == 1], bins=20, alpha=0.5, ec='black', density=True)  # female
                    axs1[i, j].set_title('Feature %d' % k)
                    k += 1
        fig1.tight_layout()
        plt.show()