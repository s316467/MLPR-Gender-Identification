import ReadData
import numpy as np
import matplotlib.pyplot as plt
import FeatureAnalysis
import Dim_reduction

if __name__ == '__main__':
    
    DT, LT = ReadData.read_data_training('./Dataset/Train.txt')
    DE, LE = ReadData.read_data_evaluation('./Dataset/Test.txt')

    ### FEATURES ANALYSIS ###
    #########################
    # Features analysis - overall statistics
    mu = np.mean(DT, axis=1).reshape(-1, 1)
    std = np.std(DT, axis=1).reshape(-1, 1)


    FeatureAnalysis.plot_hist(DT,LT)
    FeatureAnalysis.plot_hist(DT,LT, preproc='znorm')

    DTz = FeatureAnalysis.znorm_features(DT)
    FeatureAnalysis.heatmap(DT, LT, 'Features correlation (no preprocessing)')
    FeatureAnalysis.heatmap(DTz, LT, 'Features correlation (z-normalized features)')

    m, t  = Dim_reduction.kfold_PCA( D=DT, k=3, threshold=0.95, show=True) 
    
    
    

    






    
   
