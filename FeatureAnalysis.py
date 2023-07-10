import Constants
import numpy as np
from scipy.stats import norm
import seaborn
import matplotlib.pyplot as plt

def znorm_features(DT):
        DTmean = DT.mean(axis=1).reshape(-1, 1)
        DTstdDev = DT.std(axis=1).reshape(-1, 1)
        ZnormDT = (DT - DTmean) / DTstdDev
        return ZnormDT

def heatmap(DT, LT, title):
        fig, axs = plt.subplots(1, 3)
        #fig.suptitle(title)
        seaborn.heatmap(np.corrcoef(DT), linewidth=0.2, cmap="Greys", square=True, cbar=False, ax=axs[0])
        seaborn.heatmap(np.corrcoef(DT[:, LT == 0]), linewidth=0.2, cmap="Blues", square=True, cbar=False, ax=axs[1])
        seaborn.heatmap(np.corrcoef(DT[:, LT == 1]), linewidth=0.2, cmap="Oranges", square=True, cbar=False, ax=axs[2])
        plt.show()

def plot_hist(DT, LT, preproc='raw'):
        fig, axs = plt.subplots(3, 4)
        k = 0
        if preproc == 'raw':
            fig.suptitle('No preprocessing')

            for i in range(3):
                for j in range(4):
                    axs[i, j].hist(DT[k, LT == 0], bins=20, alpha=0.5, ec='black', density=True)  
                    axs[i, j].hist(DT[k, LT == 1], bins=20, alpha=0.5, ec='black', density=True)  
                    axs[i, j].set_title('Feature %d' % k)
                    k += 1
        elif preproc == 'znorm':
            DT_znorm = znorm_features(DT)
            fig.suptitle('Z-normalization preprocessing')

            for i in range(3):
                for j in range(4):
                    axs[i, j].hist(DT_znorm[k, LT == 0], bins=20, alpha=0.5, ec='black', density=True)  
                    axs[i, j].hist(DT_znorm[k, LT == 1], bins=20, alpha=0.5, ec='black', density=True) 
                    axs[i, j].set_title('Feature %d' % k)
                    k += 1
        fig.tight_layout()
        plt.show()

