import ReadData
import numpy as np
import matplotlib.pyplot as plt
import FeatureAnalysis as FA
import Dim_reduction as DimRed
import MVG_Clf
import TiedGaussian_Clf
import NaiveBayes_Clf
import LLR_Clf
import QLR_Clf
import SVM_Clf
import GMM_Clf
import Evaluation

if __name__ == '__main__':
    
    DT, LT = ReadData.read_data_training('Dataset/Train.txt')
    DE, LE = ReadData.read_data_evaluation('Dataset/Test.txt')

    # ########################
    # ## FEATURES ANALYSIS ###
    # ########################

    # # PLOTTING HIST OF RAW FEATURES
    # FA.plot_features_hist(DT, LT, 'raw', True)

    # # PLOTTING HIST OF ZNORM FEATURES
    # FA.plot_features_hist(DT, LT, 'znorm', True)

    # # PLOTTING HIST OF ZNORM+GAU FEATURES
    # FA.plot_features_hist(DT, LT, 'zg', True)

    # # PCA K-FOLD
    # m, t = DimRed.kfold_PCA(D=DT, k=5, threshold=0.95, show=True)
    
    # ###########################
    # ## GENERATIVE GAUSSIANS ###
    # ###########################

    # # MVG FULL - TIED - NAIVE -> RAW Features
    
    # dim_red = None

    # print('R: MVG Classifier\nPreprocessing: raw\nDim. Reduction: %s\nValidation: k-fold' % dim_red)
    # _, mindcf_MVG_raw_noPCA = Evaluation.kfold_cross_validation(MVG_Clf.MVG(), DT, LT, k=5, preproc='raw', dimred=dim_red,  iprint=True)
    
    # print('R: MVG-Tied Classifier\nPreprocessing: raw\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Tied_raw_noPCA = Evaluation.kfold_cross_validation(TiedGaussian_Clf.TiedG(), DT, LT, k=5, preproc='raw', dimred=dim_red, iprint=True)

    # print('R: Naive Bayes Classifier\nPreprocessing: raw\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Naive_raw_noPCA = Evaluation.kfold_cross_validation(NaiveBayes_Clf.NaiveBayes(), DT, LT, k=5, preproc='raw', dimred=dim_red, iprint=True)

    # dim_red = {'type': 'pca', 'm': 11}

    # print('R: MVG Classifier\nPreprocessing: raw\nDim. Reduction: %s\nValidation: k-fold' % dim_red)
    # _, mindcf_MVG_raw_11PCA = Evaluation.kfold_cross_validation(MVG_Clf.MVG(), DT, LT, k=5, preproc='raw', dimred=dim_red,  iprint=True)
    
    # print('R: MVG-Tied Classifier\nPreprocessing: raw\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Tied_raw_11PCA = Evaluation.kfold_cross_validation(TiedGaussian_Clf.TiedG(), DT, LT, k=5, preproc='raw', dimred=dim_red, iprint=True)

    # print('R: Naive Bayes Classifier\nPreprocessing: raw\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Naive_raw_11PCA = Evaluation.kfold_cross_validation(NaiveBayes_Clf.NaiveBayes(), DT, LT, k=5, preproc='raw', dimred=dim_red, iprint=True)

    # dim_red = {'type': 'pca', 'm': 10}

    # print('R: MVG Classifier\nPreprocessing: raw\nDim. Reduction: %s\nValidation: k-fold' % dim_red)
    # _, mindcf_MVG_raw_10PCA = Evaluation.kfold_cross_validation(MVG_Clf.MVG(), DT, LT, k=5, preproc='raw', dimred=dim_red,  iprint=True)
    
    # print('R: MVG-Tied Classifier\nPreprocessing: raw\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Tied_raw_10PCA = Evaluation.kfold_cross_validation(TiedGaussian_Clf.TiedG(), DT, LT, k=5, preproc='raw', dimred=dim_red, iprint=True)

    # print('R: Naive Bayes Classifier\nPreprocessing: raw\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Naive_raw_10PCA = Evaluation.kfold_cross_validation(NaiveBayes_Clf.NaiveBayes(), DT, LT, k=5, preproc='raw', dimred=dim_red, iprint=True)
    
    # mindcf_MVG_raw = [mindcf_MVG_raw_noPCA, mindcf_MVG_raw_11PCA, mindcf_MVG_raw_10PCA]
    # mindcf_Tied_raw = [mindcf_Tied_raw_noPCA,mindcf_Tied_raw_11PCA,mindcf_Tied_raw_10PCA]
    # mindcf_Naive_raw = [mindcf_Naive_raw_noPCA,mindcf_Naive_raw_11PCA, mindcf_Naive_raw_10PCA]

    # Evaluation.plot_gaussian_models(mindcf_MVG_raw,mindcf_Tied_raw, mindcf_Naive_raw )

    # # MVG FULL - TIED - NAIVE -> Znorm Features
    
    # dim_red = None

    # print('R: MVG Classifier\nPreprocessing: znorm\nDim. Reduction: %s\nValidation: k-fold' % dim_red)
    # _, mindcf_MVG_znorm_noPCA = Evaluation.kfold_cross_validation(MVG_Clf.MVG(), DT, LT, k=5, preproc='znorm', dimred=dim_red,  iprint=True)
    
    # print('R: MVG-Tied Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Tied_znorm_noPCA = Evaluation.kfold_cross_validation(TiedGaussian_Clf.TiedG(), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # print('R: Naive Bayes Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Naive_znorm_noPCA = Evaluation.kfold_cross_validation(NaiveBayes_Clf.NaiveBayes(), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # dim_red = {'type': 'pca', 'm': 11}

    # print('R: MVG Classifier\nPreprocessing: znorm\nDim. Reduction: %s\nValidation: k-fold' % dim_red)
    # _, mindcf_MVG_znorm_11PCA = Evaluation.kfold_cross_validation(MVG_Clf.MVG(), DT, LT, k=5, preproc='znorm', dimred=dim_red,  iprint=True)
    
    # print('R: MVG-Tied Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Tied_znorm_11PCA = Evaluation.kfold_cross_validation(TiedGaussian_Clf.TiedG(), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # print('R: Naive Bayes Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Naive_znorm_11PCA = Evaluation.kfold_cross_validation(NaiveBayes_Clf.NaiveBayes(), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # dim_red = {'type': 'pca', 'm': 10}

    # print('R: MVG Classifier\nPreprocessing: znorm\nDim. Reduction: %s\nValidation: k-fold' % dim_red)
    # _, mindcf_MVG_znorm_10PCA = Evaluation.kfold_cross_validation(MVG_Clf.MVG(), DT, LT, k=5, preproc='znorm', dimred=dim_red,  iprint=True)
    
    # print('R: MVG-Tied Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Tied_znorm_10PCA = Evaluation.kfold_cross_validation(TiedGaussian_Clf.TiedG(), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # print('R: Naive Bayes Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Naive_znorm_10PCA = Evaluation.kfold_cross_validation(NaiveBayes_Clf.NaiveBayes(), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    
    # mindcf_MVG_znorm = [mindcf_MVG_znorm_noPCA, mindcf_MVG_znorm_11PCA, mindcf_MVG_znorm_10PCA]
    # mindcf_Tied_znorm = [mindcf_Tied_znorm_noPCA,mindcf_Tied_znorm_11PCA,mindcf_Tied_znorm_10PCA]
    # mindcf_Naive_znorm = [mindcf_Naive_znorm_noPCA,mindcf_Naive_znorm_11PCA, mindcf_Naive_znorm_10PCA]

    # Evaluation.plot_gaussian_models(mindcf_MVG_znorm,mindcf_Tied_znorm, mindcf_Naive_znorm )
    
    
    # # MVG FULL - TIED - NAIVE -> Znorm + gauss Features

    
    # dim_red = None

    # print('R: MVG Classifier\nPreprocessing: zg\nDim. Reduction: %s\nValidation: k-fold' % dim_red)
    # _, mindcf_MVG_zg_noPCA = Evaluation.kfold_cross_validation(MVG_Clf.MVG(), DT, LT, k=5, preproc='zg', dimred=dim_red,  iprint=True)
    
    # print('R: MVG-Tied Classifier\nPreprocessing: zg\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Tied_zg_noPCA = Evaluation.kfold_cross_validation(TiedGaussian_Clf.TiedG(), DT, LT, k=5, preproc='zg', dimred=dim_red, iprint=True)

    # print('R: Naive Bayes Classifier\nPreprocessing: zg\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Naive_zg_noPCA = Evaluation.kfold_cross_validation(NaiveBayes_Clf.NaiveBayes(), DT, LT, k=5, preproc='zg', dimred=dim_red, iprint=True)

    # dim_red = {'type': 'pca', 'm': 11}

    # print('R: MVG Classifier\nPreprocessing: zg\nDim. Reduction: %s\nValidation: k-fold' % dim_red)
    # _, mindcf_MVG_zg_11PCA = Evaluation.kfold_cross_validation(MVG_Clf.MVG(), DT, LT, k=5, preproc='zg', dimred=dim_red,  iprint=True)
    
    # print('R: MVG-Tied Classifier\nPreprocessing: zg\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Tied_zg_11PCA = Evaluation.kfold_cross_validation(TiedGaussian_Clf.TiedG(), DT, LT, k=5, preproc='zg', dimred=dim_red, iprint=True)

    # print('R: Naive Bayes Classifier\nPreprocessing: zg\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Naive_zg_11PCA = Evaluation.kfold_cross_validation(NaiveBayes_Clf.NaiveBayes(), DT, LT, k=5, preproc='zg', dimred=dim_red, iprint=True)

    # dim_red = {'type': 'pca', 'm': 10}

    # print('R: MVG Classifier\nPreprocessing: zg\nDim. Reduction: %s\nValidation: k-fold' % dim_red)
    # _, mindcf_MVG_zg_10PCA = Evaluation.kfold_cross_validation(MVG_Clf.MVG(), DT, LT, k=5, preproc='zg', dimred=dim_red,  iprint=True)
    
    # print('R: MVG-Tied Classifier\nPreprocessing: zg\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Tied_zg_10PCA = Evaluation.kfold_cross_validation(TiedGaussian_Clf.TiedG(), DT, LT, k=5, preproc='zg', dimred=dim_red, iprint=True)

    # print('R: Naive Bayes Classifier\nPreprocessing: zg\nDim. Reduction: %s\n' % dim_red)
    # _, mindcf_Naive_zg_10PCA = Evaluation.kfold_cross_validation(NaiveBayes_Clf.NaiveBayes(), DT, LT, k=5, preproc='zg', dimred=dim_red, iprint=True)
    
    # mindcf_MVG_zg = [mindcf_MVG_zg_noPCA, mindcf_MVG_zg_11PCA, mindcf_MVG_zg_10PCA]
    # mindcf_Tied_zg = [mindcf_Tied_zg_noPCA,mindcf_Tied_zg_11PCA,mindcf_Tied_zg_10PCA]
    # mindcf_Naive_zg = [mindcf_Naive_zg_noPCA,mindcf_Naive_zg_11PCA, mindcf_Naive_zg_10PCA]

    # Evaluation.plot_gaussian_models(mindcf_MVG_zg,mindcf_Tied_zg, mindcf_Naive_zg )

    
 
    # ##################################
    # ### LINEAR LOGISTIC REGRESSION ###
    # ##################################
    
    # #AT FIRST ANALYZE THE PERFORMANCE OF NON PRIOR-WIGHTED LLR FOR DIFFERENT VALUES OF LAMBDA ON ZNORM FEATURES

    
    # dim_red = None 
    # lambdas = [1.E-6, 1.E-5, 1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10, 100, 1000, 10000, 100000]
    # mindcf_LLR_noPCA_01p = []
    # mindcf_LLR_11PCA_01p = []
    # mindcf_LLR_10PCA_01p = []

    # mindcf_LLR_noPCA_05p = []
    # mindcf_LLR_11PCA_05p = []
    # mindcf_LLR_10PCA_05p = []

    # mindcf_LLR_noPCA_09p = []
    # mindcf_LLR_11PCA_09p = []
    # mindcf_LLR_10PCA_09p = []

    # for l in lambdas:
    #     print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    #     _ , mindcf = Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_LLR_noPCA_01p.append(mindcf[0])
    #     mindcf_LLR_noPCA_05p.append(mindcf[1])
    #     mindcf_LLR_noPCA_09p.append(mindcf[2])

 
    # dim_red = {'type': 'pca', 'm': 11}
    # for l in lambdas:
    #     print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    #     _ , mindcf = Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_LLR_11PCA_01p.append(mindcf[0])
    #     mindcf_LLR_11PCA_05p.append(mindcf[1])
    #     mindcf_LLR_11PCA_09p.append(mindcf[2])
    
    # dim_red = {'type': 'pca', 'm': 10}
    # for l in lambdas:
    #     print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    #     _ , mindcf = Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_LLR_10PCA_01p.append(mindcf[0])
    #     mindcf_LLR_10PCA_05p.append(mindcf[1])
    #     mindcf_LLR_10PCA_09p.append(mindcf[2])


    # mindcf_LLR_noPCA = [ mindcf_LLR_noPCA_01p, mindcf_LLR_noPCA_05p, mindcf_LLR_noPCA_09p]
    # mindcf_LLR_11PCA = [ mindcf_LLR_11PCA_01p, mindcf_LLR_11PCA_05p, mindcf_LLR_11PCA_09p]
    # mindcf_LLR_10PCA = [ mindcf_LLR_10PCA_01p, mindcf_LLR_10PCA_05p, mindcf_LLR_10PCA_09p]

    # Evaluation.plot_lambda_minDCF_LLR(lambdas, mindcf_LLR_noPCA, mindcf_LLR_11PCA, mindcf_LLR_10PCA )

    # #From the precedent plot we noticed that we need to choose a value less than 10^-3 to obtain good result in terms of minDCF
    # #So here it is a more in depth analysis of the minDCF

    # #NO PCA
    # dim_red = None
    # #NON PRIOR
    # l = 10**-6
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # #PRIOR FOR TARGET APPLICATION (0.5)
    # l = 10**-6
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    
    

    # #11 PCA
    # dim_red = {'type': 'pca', 'm': 11}
    
    # #NON PRIOR
    # l = 10**-6
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # #PRIOR FOR TARGET APPLICATION (0.5)
    # l = 10**-6
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    
    # #10 PCA
    # dim_red = {'type': 'pca', 'm': 10}
    
    # #NON PRIOR
    # l = 10**-6
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # #PRIOR FOR TARGET APPLICATION (0.5)
    # l = 10**-6
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    
    

    # # #####################################
    # # ### QUADRATIC LOGISTIC REGRESSION ###
    # # #####################################

    # #AT FIRST ANALYZE THE PERFORMANCE OF NON PRIOR-WIGHTED LLR FOR DIFFERENT VALUES OF LAMBDA ON ZNORM FEATURES

    
    # dim_red = None 
    # lambdas = [1.E-6, 1.E-5, 1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10, 100, 1000, 10000, 100000]
    # mindcf_QLR_noPCA_01p = []
    # mindcf_QLR_11PCA_01p = []
    # mindcf_QLR_10PCA_01p = []

    # mindcf_QLR_noPCA_05p = []
    # mindcf_QLR_11PCA_05p = []
    # mindcf_QLR_10PCA_05p = []

    # mindcf_QLR_noPCA_09p = []
    # mindcf_QLR_11PCA_09p = []
    # mindcf_QLR_10PCA_09p = []

    # for l in lambdas:
    #     print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    #     _ , mindcf = Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_QLR_noPCA_01p.append(mindcf[0])
    #     mindcf_QLR_noPCA_05p.append(mindcf[1])
    #     mindcf_QLR_noPCA_09p.append(mindcf[2])

 
    # dim_red = {'type': 'pca', 'm': 11}
    # for l in lambdas:
    #     print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    #     _ , mindcf = Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_QLR_11PCA_01p.append(mindcf[0])
    #     mindcf_QLR_11PCA_05p.append(mindcf[1])
    #     mindcf_QLR_11PCA_09p.append(mindcf[2])
    
    # dim_red = {'type': 'pca', 'm': 10}
    # for l in lambdas:
    #     print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    #     _ , mindcf = Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_QLR_10PCA_01p.append(mindcf[0])
    #     mindcf_QLR_10PCA_05p.append(mindcf[1])
    #     mindcf_QLR_10PCA_09p.append(mindcf[2])


    # mindcf_QLR_noPCA = [ mindcf_QLR_noPCA_01p, mindcf_QLR_noPCA_05p, mindcf_QLR_noPCA_09p]
    # mindcf_QLR_11PCA = [ mindcf_QLR_11PCA_01p, mindcf_QLR_11PCA_05p, mindcf_QLR_11PCA_09p]
    # mindcf_QLR_10PCA = [ mindcf_QLR_10PCA_01p, mindcf_QLR_10PCA_05p, mindcf_QLR_10PCA_09p]

    # Evaluation.plot_lambda_minDCF_LLR(lambdas, mindcf_QLR_noPCA, mindcf_QLR_11PCA, mindcf_QLR_10PCA )
    

    # #From the precedent plot we noticed that we need to choose a value less than 10^-3 to obtain good result in terms of minDCF
    # #So here it is a more in depth analysis of the minDCF

    # #NO PCA
    
    # dim_red = None
    
    # #NON PRIOR
    # l = 10**-6
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # #PRIOR FOR TARGET APPLICATION (0.5)
    # l = 10**-6
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    
    

    # #11 PCA
    # dim_red = {'type': 'pca', 'm': 11}
    
    # #NON PRIOR
    # l = 10**-6
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # #PRIOR FOR TARGET APPLICATION (0.5)
    # l = 10**-6
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    
    # #10 PCA
    # dim_red = {'type': 'pca', 'm': 10}
    
    # #NON PRIOR
    # l = 10**-6
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=False), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # #PRIOR FOR TARGET APPLICATION (0.5)
    # l = 10**-6
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-5
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-4
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # l = 10**-3
    # print('R: Quadratic Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(QLR_Clf.QuadraticLogisticRegression(l, prior_weighted=True, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    
    
    # ##################
    # ### LINEAR SVM ###
    # ##################

    

    # dim_red = None #{'type': 'pca', 'm': 11}
    # C = [1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10, 100]
    # hparams = {'K': 0, 'C': 0}

    # mindcf_linearSVM_noPCA_01p = []
    # mindcf_linearSVM_noPCA_05p = []
    # mindcf_linearSVM_noPCA_09p = []

    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, None), DT, LT, k=5,preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_linearSVM_noPCA_01p.append(mindcf[0])
    #     mindcf_linearSVM_noPCA_05p.append(mindcf[1])
    #     mindcf_linearSVM_noPCA_09p.append(mindcf[2])

    # dim_red = {'type': 'pca', 'm': 11}
    # mindcf_linearSVM_11PCA_01p = []
    # mindcf_linearSVM_11PCA_05p = []
    # mindcf_linearSVM_11PCA_09p = []
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, None), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_linearSVM_11PCA_01p.append(mindcf[0])
    #     mindcf_linearSVM_11PCA_05p.append(mindcf[1])
    #     mindcf_linearSVM_11PCA_09p.append(mindcf[2])

    # dim_red = {'type': 'pca', 'm': 10}
    # mindcf_linearSVM_10PCA_01p = []
    # mindcf_linearSVM_10PCA_05p = []
    # mindcf_linearSVM_10PCA_09p = []
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, None), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_linearSVM_10PCA_01p.append(mindcf[0])
    #     mindcf_linearSVM_10PCA_05p.append(mindcf[1])
    #     mindcf_linearSVM_10PCA_09p.append(mindcf[2])

    # mindcf_linearSVM_noPCA = [ mindcf_linearSVM_noPCA_01p, mindcf_linearSVM_noPCA_05p, mindcf_linearSVM_noPCA_09p]
    # mindcf_linearSVM_11PCA = [ mindcf_linearSVM_11PCA_01p, mindcf_linearSVM_11PCA_05p, mindcf_linearSVM_11PCA_09p]
    # mindcf_linearSVM_10PCA = [ mindcf_linearSVM_10PCA_01p, mindcf_linearSVM_10PCA_05p, mindcf_linearSVM_10PCA_09p]

    # Evaluation.plot_lambda_minDCF_LinearSVM(C, mindcf_linearSVM_noPCA, mindcf_linearSVM_11PCA, mindcf_linearSVM_10PCA)

    # #After the plot we saw that for c>=10^-1 we obtain the best value for minDCF
    # #We will analyze with znorm preprocessed feature

    # #No PCA
    # dim_red = None

    # #Non Prior
    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 10**-1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, None), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, None), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # #Prior
    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 10**-1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # #11 PCA
    # dim_red = {'type': 'pca', 'm': 11}
    # #Non Prior
    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 10**-1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, None), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, None), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # #Prior
    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 10**-1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # #10 PCA
    # dim_red = {'type': 'pca', 'm': 10}
    # #Non Prior
    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 10**-1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, None), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, None), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # #Prior
    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 10**-1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)

    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 1}
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, prior=0.5), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    

    # #############################
    # ### POLYNOMIAL KERNEL SVM ###
    # #############################

    

    # hparams = {'K': 1, 'eps': 1, 'gamma': 1, 'C': 0, 'c': 0, 'd': 2}
    # dim_red = None
    # C = [1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10, 100]

    # mindcf_PSVM_noPCA_01p = []
    # mindcf_PSVM_noPCA_05p = []
    # mindcf_PSVM_noPCA_09p = []

    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)
    #     mindcf_PSVM_noPCA_01p.append(mindcf[0])
    #     mindcf_PSVM_noPCA_05p.append(mindcf[1])
    #     mindcf_PSVM_noPCA_09p.append(mindcf[2])

    # dim_red = {'type': 'pca', 'm': 11}
    # mindcf_PSVM_11PCA_01p = []
    # mindcf_PSVM_11PCA_05p = []
    # mindcf_PSVM_11PCA_09p = []
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5, preproc='raw', dimred=dim_red, iprint=True)
    #     mindcf_PSVM_11PCA_01p.append(mindcf[0])
    #     mindcf_PSVM_11PCA_05p.append(mindcf[1])
    #     mindcf_PSVM_11PCA_09p.append(mindcf[2])

    # dim_red = {'type': 'pca', 'm': 10}
    # mindcf_PSVM_10PCA_01p = []
    # mindcf_PSVM_10PCA_05p = []
    # mindcf_PSVM_10PCA_09p = []
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5, preproc='raw', dimred=dim_red, iprint=True)
    #     mindcf_PSVM_10PCA_01p.append(mindcf[0])
    #     mindcf_PSVM_10PCA_05p.append(mindcf[1])
    #     mindcf_PSVM_10PCA_09p.append(mindcf[2])

    # mindcf_PSVM_noPCA = [ mindcf_PSVM_noPCA_01p, mindcf_PSVM_noPCA_05p, mindcf_PSVM_noPCA_09p]
    # mindcf_PSVM_11PCA = [ mindcf_PSVM_11PCA_01p, mindcf_PSVM_11PCA_05p, mindcf_PSVM_11PCA_09p]
    # mindcf_PSVM_10PCA = [ mindcf_PSVM_10PCA_01p, mindcf_PSVM_10PCA_05p, mindcf_PSVM_10PCA_09p]

    # Evaluation.plot_lambda_minDCF_LinearSVM(C, mindcf_PSVM_noPCA, mindcf_PSVM_11PCA, mindcf_PSVM_10PCA)


    # #After the plot we choose c=10^-4 to show better results varying c and d
    # #No PCA
    # dim_red = None
    # #No prior
    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 0, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 3}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # #Prior 0.5
    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 0, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial', prior=0.5), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial', prior=0.5), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 3}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial', prior=0.5), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # #11 PCA
    # dim_red = {'type': 'pca', 'm': 11}
    # #No prior
    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 0, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 3}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # #Prior 0.5
    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 0, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial', prior=0.5), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial', prior=0.5), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 3}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial', prior=0.5), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)


    # #10 PCA
    # dim_red = {'type': 'pca', 'm': 10}
    # #No prior
    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 0, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 3}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial'), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # #Prior 0.5
    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 0, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial', prior=0.5), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 2}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial', prior=0.5), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10**-4, 'c': 1, 'd': 3}
    # print('R: SVM Poly\nPreprocessing: raw\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='Polynomial', prior=0.5), DT, LT, k=5,preproc='raw', dimred=dim_red, iprint=True)
    
    # #####################
    # ## RBF KERNEL SVM ###
    # #####################
    # # Testing various gammas => 0.01 0.1 1
    
    # hparams = {'K': 1, 'eps': 1, 'gamma': 0.01, 'C': 1, 'd': 1}
    # dim_red = None #{'type': 'pca', 'm': 11}
    # C = [1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10, 100]
    # mindcf_RBFSVM_noPCA_01p_001g = []
    # mindcf_RBFSVM_noPCA_05p_001g = []
    # mindcf_RBFSVM_noPCA_09p_001g = []
    
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM RBF\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='RBF'), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_RBFSVM_noPCA_01p_001g.append(mindcf[0])
    #     mindcf_RBFSVM_noPCA_05p_001g.append(mindcf[1])
    #     mindcf_RBFSVM_noPCA_09p_001g.append(mindcf[2])

    # mindcf_RBFSVM_noPCA_001g = [ mindcf_RBFSVM_noPCA_01p_001g, mindcf_RBFSVM_noPCA_05p_001g, mindcf_RBFSVM_noPCA_09p_001g]

    # hparams['gamma'] = 0.1
    # mindcf_RBFSVM_noPCA_01p_01g = []
    # mindcf_RBFSVM_noPCA_05p_01g = []
    # mindcf_RBFSVM_noPCA_09p_01g = []
    
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM RBF\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='RBF'), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_RBFSVM_noPCA_01p_01g.append(mindcf[0])
    #     mindcf_RBFSVM_noPCA_05p_01g.append(mindcf[1])
    #     mindcf_RBFSVM_noPCA_09p_01g.append(mindcf[2])

    # mindcf_RBFSVM_noPCA_01g = [ mindcf_RBFSVM_noPCA_01p_01g, mindcf_RBFSVM_noPCA_05p_01g, mindcf_RBFSVM_noPCA_09p_01g]

    # hparams['gamma'] = 1
    # mindcf_RBFSVM_noPCA_01p_1g = []
    # mindcf_RBFSVM_noPCA_05p_1g = []
    # mindcf_RBFSVM_noPCA_09p_1g = []
    
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM RBF\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='RBF'), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_RBFSVM_noPCA_01p_1g.append(mindcf[0])
    #     mindcf_RBFSVM_noPCA_05p_1g.append(mindcf[1])
    #     mindcf_RBFSVM_noPCA_09p_1g.append(mindcf[2])

    # mindcf_RBFSVM_noPCA_1g = [ mindcf_RBFSVM_noPCA_01p_1g, mindcf_RBFSVM_noPCA_05p_1g, mindcf_RBFSVM_noPCA_09p_1g]

    # Evaluation.plot_lambda_minDCF_RBFSVM(C, mindcf_RBFSVM_noPCA_001g, mindcf_RBFSVM_noPCA_01g, mindcf_RBFSVM_noPCA_1g)

    

    # #SVM RBF with PCA m=11
    
    # hparams = {'K': 1, 'eps': 1, 'gamma': 0.01, 'C': 1, 'c': 0, 'd': 1}
    # dim_red = {'type': 'pca', 'm': 11}
    # C = [1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10, 100]
    # mindcf_RBFSVM_11PCA_01p_001g = []
    # mindcf_RBFSVM_11PCA_05p_001g = []
    # mindcf_RBFSVM_11PCA_09p_001g = []
    
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM RBF\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='RBF'), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_RBFSVM_11PCA_01p_001g.append(mindcf[0])
    #     mindcf_RBFSVM_11PCA_05p_001g.append(mindcf[1])
    #     mindcf_RBFSVM_11PCA_09p_001g.append(mindcf[2])

    # mindcf_RBFSVM_11PCA_001g = [ mindcf_RBFSVM_11PCA_01p_001g, mindcf_RBFSVM_11PCA_05p_001g, mindcf_RBFSVM_11PCA_09p_001g]

    # hparams['gamma'] = 0.1
    # mindcf_RBFSVM_11PCA_01p_01g = []
    # mindcf_RBFSVM_11PCA_05p_01g = []
    # mindcf_RBFSVM_11PCA_09p_01g = []
    
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM RBF\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='RBF'), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_RBFSVM_11PCA_01p_01g.append(mindcf[0])
    #     mindcf_RBFSVM_11PCA_05p_01g.append(mindcf[1])
    #     mindcf_RBFSVM_11PCA_09p_01g.append(mindcf[2])

    # mindcf_RBFSVM_11PCA_01g = [ mindcf_RBFSVM_11PCA_01p_01g, mindcf_RBFSVM_11PCA_05p_01g, mindcf_RBFSVM_11PCA_09p_01g]

    # hparams['gamma'] = 1
    # mindcf_RBFSVM_11PCA_01p_1g = []
    # mindcf_RBFSVM_11PCA_05p_1g = []
    # mindcf_RBFSVM_11PCA_09p_1g = []
    
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM RBF\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='RBF'), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_RBFSVM_11PCA_01p_1g.append(mindcf[0])
    #     mindcf_RBFSVM_11PCA_05p_1g.append(mindcf[1])
    #     mindcf_RBFSVM_11PCA_09p_1g.append(mindcf[2])

    # mindcf_RBFSVM_11PCA_1g = [ mindcf_RBFSVM_11PCA_01p_1g, mindcf_RBFSVM_11PCA_05p_1g, mindcf_RBFSVM_11PCA_09p_1g]

    # Evaluation.plot_lambda_minDCF_RBFSVM(C, mindcf_RBFSVM_11PCA_001g, mindcf_RBFSVM_11PCA_01g, mindcf_RBFSVM_11PCA_1g)
    

    # #SVM RBF with PCA m=10
    
    # hparams = {'K': 1, 'eps': 1, 'gamma': 0.01, 'C': 1, 'c': 0, 'd': 1}
    # dim_red = {'type': 'pca', 'm': 11}
    # C = [1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10, 100]
    # mindcf_RBFSVM_10PCA_01p_001g = []
    # mindcf_RBFSVM_10PCA_05p_001g = []
    # mindcf_RBFSVM_10PCA_09p_001g = []
    
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM RBF\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='RBF'), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_RBFSVM_10PCA_01p_001g.append(mindcf[0])
    #     mindcf_RBFSVM_10PCA_05p_001g.append(mindcf[1])
    #     mindcf_RBFSVM_10PCA_09p_001g.append(mindcf[2])

    # mindcf_RBFSVM_10PCA_001g = [ mindcf_RBFSVM_10PCA_01p_001g, mindcf_RBFSVM_10PCA_05p_001g, mindcf_RBFSVM_10PCA_09p_001g]

    # hparams['gamma'] = 0.1
    # mindcf_RBFSVM_10PCA_01p_01g = []
    # mindcf_RBFSVM_10PCA_05p_01g = []
    # mindcf_RBFSVM_10PCA_09p_01g = []
    
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM RBF\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='RBF'), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_RBFSVM_10PCA_01p_01g.append(mindcf[0])
    #     mindcf_RBFSVM_10PCA_05p_01g.append(mindcf[1])
    #     mindcf_RBFSVM_10PCA_09p_01g.append(mindcf[2])

    # mindcf_RBFSVM_10PCA_01g = [ mindcf_RBFSVM_10PCA_01p_01g, mindcf_RBFSVM_10PCA_05p_01g, mindcf_RBFSVM_10PCA_09p_01g]

    # hparams['gamma'] = 1
    # mindcf_RBFSVM_10PCA_01p_1g = []
    # mindcf_RBFSVM_10PCA_05p_1g = []
    # mindcf_RBFSVM_10PCA_09p_1g = []
    
    # for c in C:
    #     hparams['C'] = c
    #     print('R: SVM RBF\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    #     _, mindcf = Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='RBF'), DT, LT, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    #     mindcf_RBFSVM_10PCA_01p_1g.append(mindcf[0])
    #     mindcf_RBFSVM_10PCA_05p_1g.append(mindcf[1])
    #     mindcf_RBFSVM_10PCA_09p_1g.append(mindcf[2])

    # mindcf_RBFSVM_10PCA_1g = [ mindcf_RBFSVM_10PCA_01p_1g, mindcf_RBFSVM_10PCA_05p_1g, mindcf_RBFSVM_10PCA_09p_1g]
    
    # Evaluation.plot_lambda_minDCF_RBFSVM(C, mindcf_RBFSVM_10PCA_001g, mindcf_RBFSVM_10PCA_01g, mindcf_RBFSVM_10PCA_1g)
    
    
    # ############
    # #### GMM ###
    # ############

    # Evaluation.BinaryModelEvaluator().plot_histogramGMM(DT, LT) 

    
    # #GMM FULL
    # cov = 'Full'
    # comp = [1, 2, 4, 8, 16]
    # dim_red = None #{'type': 'pca', 'm': 10}
    # mindcf_GMMFull_noPCA_01p_znorm = []
    # mindcf_GMMFull_noPCA_05p_znorm = []
    # mindcf_GMMFull_noPCA_09p_znorm = []
    # mindcf_GMMFull_noPCA_01p_zg = []
    # mindcf_GMMFull_noPCA_05p_zg = []
    # mindcf_GMMFull_noPCA_09p_zg = []
    
    # for c in comp:
    #     print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (c, 'Full', dim_red))
    #     _, mindcf = Evaluation.kfold_cross_validation(GMM_Clf.GMM(alpha=0.1, nComponents=c, psi=0.01, covType=cov), DT, LT, k=5, preproc='znorm',dimred=dim_red)
    #     mindcf_GMMFull_noPCA_01p_znorm.append(mindcf[0])
    #     mindcf_GMMFull_noPCA_05p_znorm.append(mindcf[1])
    #     mindcf_GMMFull_noPCA_09p_znorm.append(mindcf[2])
    #     print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: zg\nDim. Reduction: %s\n' % (c, 'Full', dim_red))
    #     _, mindcf = Evaluation.kfold_cross_validation(GMM_Clf.GMM(alpha=0.1, nComponents=c, psi=0.01, covType=cov), DT, LT, k=5, preproc='zg',dimred=dim_red)
    #     mindcf_GMMFull_noPCA_01p_zg.append(mindcf[0])
    #     mindcf_GMMFull_noPCA_05p_zg.append(mindcf[1])
    #     mindcf_GMMFull_noPCA_09p_zg.append(mindcf[2])

    # mindcf_GMMFull_noPCA_05p = [mindcf_GMMFull_noPCA_05p_znorm, mindcf_GMMFull_noPCA_05p_zg]

    # cov = 'Tied'
    # mindcf_GMMTied_noPCA_01p_znorm = []
    # mindcf_GMMTied_noPCA_05p_znorm = []
    # mindcf_GMMTied_noPCA_09p_znorm = []
    # mindcf_GMMTied_noPCA_01p_zg = []
    # mindcf_GMMTied_noPCA_05p_zg = []
    # mindcf_GMMTied_noPCA_09p_zg = []
    
    # for c in comp:
    #     print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (c, 'Tied', dim_red))
    #     _, mindcf = Evaluation.kfold_cross_validation(GMM_Clf.GMM(alpha=0.1, nComponents=c, psi=0.01, covType=cov), DT, LT, k=5, preproc='znorm',dimred=dim_red)
    #     mindcf_GMMTied_noPCA_01p_znorm.append(mindcf[0])
    #     mindcf_GMMTied_noPCA_05p_znorm.append(mindcf[1])
    #     mindcf_GMMTied_noPCA_09p_znorm.append(mindcf[2])
    #     print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: zg\nDim. Reduction: %s\n' % (c, 'Tied', dim_red))
    #     _, mindcf = Evaluation.kfold_cross_validation(GMM_Clf.GMM(alpha=0.1, nComponents=c, psi=0.01, covType=cov), DT, LT, k=5, preproc='zg',dimred=dim_red)
    #     mindcf_GMMTied_noPCA_01p_zg.append(mindcf[0])
    #     mindcf_GMMTied_noPCA_05p_zg.append(mindcf[1])
    #     mindcf_GMMTied_noPCA_09p_zg.append(mindcf[2])

    # mindcf_GMMTied_noPCA_05p = [mindcf_GMMTied_noPCA_05p_znorm, mindcf_GMMTied_noPCA_05p_zg]

    # cov = 'Diag'
    # mindcf_GMMDiag_noPCA_01p_znorm = []
    # mindcf_GMMDiag_noPCA_05p_znorm = []
    # mindcf_GMMDiag_noPCA_09p_znorm = []
    # mindcf_GMMDiag_noPCA_01p_zg = []
    # mindcf_GMMDiag_noPCA_05p_zg = []
    # mindcf_GMMDiag_noPCA_09p_zg = []
    
    # for c in comp:
    #     print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (c, 'Diag', dim_red))
    #     _, mindcf = Evaluation.kfold_cross_validation(GMM_Clf.GMM(alpha=0.1, nComponents=c, psi=0.01, covType=cov), DT, LT, k=5, preproc='znorm',dimred=dim_red)
    #     mindcf_GMMDiag_noPCA_01p_znorm.append(mindcf[0])
    #     mindcf_GMMDiag_noPCA_05p_znorm.append(mindcf[1])
    #     mindcf_GMMDiag_noPCA_09p_znorm.append(mindcf[2])
    #     print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: zg\nDim. Reduction: %s\n' % (c, 'Diag', dim_red))
    #     _, mindcf = Evaluation.kfold_cross_validation(GMM_Clf.GMM(alpha=0.1, nComponents=c, psi=0.01, covType=cov), DT, LT, k=5, preproc='zg',dimred=dim_red)
    #     mindcf_GMMDiag_noPCA_01p_zg.append(mindcf[0])
    #     mindcf_GMMDiag_noPCA_05p_zg.append(mindcf[1])
    #     mindcf_GMMDiag_noPCA_09p_zg.append(mindcf[2])

    # mindcf_GMMDiag_noPCA_05p = [mindcf_GMMDiag_noPCA_05p_znorm, mindcf_GMMDiag_noPCA_05p_zg]

    # mindcf_GMM_noPCA_05p = [mindcf_GMMFull_noPCA_05p, mindcf_GMMTied_noPCA_05p, mindcf_GMMDiag_noPCA_05p]

    # Evaluation.plot_histogramGMM(comp, mindcf_GMM_noPCA_05p)
    
    
    # ##############################
    # ### SCORE CALIBRATION PLOT ###
    # ##############################

    # fig1, axs1 = plt.subplots(2, 2)
    # fig1.set_figheight(5)
    # fig1.set_figwidth(13)
    # Evaluation.plot_Bayes_error(ax=axs1[0, 0], title='Tied Covariance',
    #                                     model=MVG_Clf.TiedG(), preproc='znorm',
    #                                     dimred=None, DT=DT, LT=LT, calibrate_scores=False)
    # Evaluation.plot_Bayes_error(ax=axs1[0, 1], title='Linear Logistic Regression',
    #                                     model=LLR_Clf.LinearLogisticRegression(lbd=10**-4, prior_weighted=True, prior=0.5), preproc='znorm',
    #                                     dimred=None, DT=DT, LT=LT, calibrate_scores=False)
    # Evaluation.plot_Bayes_error(ax=axs1[1, 0], title='RBF Kernel SVM',
    #                                     model=SVM_Clf.SVM(hparams={'K': 1, 'C': 10, 'gamma': 10**-1}, kernel = 'RBF'), preproc='znorm',
    #                                     dimred=None, DT=DT, LT=LT, calibrate_scores=False)
    # Evaluation.plot_Bayes_error(ax=axs1[1, 1], title='GMM',
    #                                     model=GMM_Clf.GMM(alpha=0.1, nComponents=4, psi=0.1, covType='Tied'), preproc='znorm',
    #                                     dimred=None, DT=DT, LT=LT, calibrate_scores=False)
    # fig1.tight_layout()
    # plt.show()

    # fig1, axs1 = plt.subplots(2, 2)
    # fig1.set_figheight(5)
    # fig1.set_figwidth(13)
    # Evaluation.plot_Bayes_error(ax=axs1[0, 0], title='Tied Covariance',
    #                                     model=MVG_Clf.TiedG(), preproc='znorm',
    #                                     dimred=None, DT=DT, LT=LT, calibrate_scores=True)
    # Evaluation.plot_Bayes_error(ax=axs1[0, 1], title='Linear Logistic Regression',
    #                                     model=LLR_Clf.LinearLogisticRegression(lbd=10**-4, prior_weighted=True, prior=0.5), preproc='znorm',
    #                                     dimred=None, DT=DT, LT=LT, calibrate_scores=True)
    # Evaluation.plot_Bayes_error(ax=axs1[1, 0], title='RBF Kernel SVM',
    #                                     model=SVM_Clf.SVM(hparams={'K': 1, 'C': 10, 'gamma': 10**-1}, kernel = 'RBF'), preproc='znorm',
    #                                     dimred=None, DT=DT, LT=LT, calibrate_scores=True)
    # Evaluation.plot_Bayes_error(ax=axs1[1, 1], title='GMM',
    #                                     model=GMM_Clf.GMM(alpha=0.1, nComponents=4, psi=0.1, covType='Tied'), preproc='znorm',
    #                                     dimred=None, DT=DT, LT=LT, calibrate_scores=True)
    # fig1.tight_layout()
    # plt.show()

    # ##########################
    # ## UNCALIBRATED SCORES ###
    # ##########################

    # # MVG TIED # best gaussian model
    # dim_red = None

    # print('R: MVG-Tied Classifier\nPreprocessing: raw\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(MVG_Clf.TiedG(), DT, LT, k=5, preproc='raw', dimred=dim_red)
    
    
    # lbd = 10**-4 # best LLR model
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(lbd, prior_weighted=True, prior=0.5),
    #                                        DT,
    #                                        LT,
    #                                        k=5,
    #                                        preproc='znorm',
    #                                        dimred=dim_red)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 10**-1, 'C': 10, 'c': 0, 'd': 1}  # best RBF K SVM model
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='RBF'),
    #                                        DT,
    #                                        LT,
    #                                        k=5,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        iprint=True)
    # hparams = {'K': 0, 'eps': 0, 'C': 1}  # best linear K SVM model
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams),
    #                                        DT,
    #                                        LT,
    #                                        k=5,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        iprint=True)
    # cov = 'Tied' # best GMM model
    # nComponents = 4
    # print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    # Evaluation.kfold_cross_validation(GMM_Clf.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
    #                                        DT,
    #                                        LT,
    #                                        k=5,
    #                                        preproc='znorm',
    #                                        dimred=dim_red)


    
    # ########################
    # ## CALIBRATED SCORES ###
    # ########################

    # # MVG TIED # best gaussian model
    # dim_red = None
    # model_evaluator = Evaluation.BinaryModelEvaluator()

    # print('R: MVG-Tied Classifier\nPreprocessing: raw\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(MVG_Clf.TiedG(), DT, LT, k=5, preproc='raw', dimred=dim_red, calibrated = True)
    
    
    # lbd = 10**-4 # best LLR model
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.kfold_cross_validation(LLR_Clf.LinearLogisticRegression(lbd, prior_weighted=True, prior=0.5),
    #                                        DT,
    #                                        LT,
    #                                        k=5,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        calibrated = True)

    # hparams = {'K': 1, 'eps': 0, 'gamma': 10**-1, 'C': 10, 'c': 0, 'd': 1}  # best RBF K SVM model
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams, kernel='RBF'),
    #                                        DT,
    #                                        LT,
    #                                        k=5,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        calibrated = True,
    #                                        iprint=True)

    # hparams = {'K': 0, 'eps': 0, 'C': 1}  # best linear K SVM model
    # Evaluation.kfold_cross_validation(SVM_Clf.SVM(hparams),
    #                                        DT,
    #                                        LT,
    #                                        k=5,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        calibrated = True,
    #                                        iprint=True)

    # cov = 'Tied' # best GMM model
    # nComponents = 4
    # print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    # Evaluation.kfold_cross_validation(GMM_Clf.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
    #                                        DT,
    #                                        LT,
    #                                        k=5,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        calibrated = True)

    # ###################################
    # ### EXPERIMENTAL RESULTS NO PCA ###
    # ###################################
    
    
    # ##########################
    # # GENERATIVE GAUSSIANS ###
    # ##########################

    # # MVG FULL

    # dim_red = None # {'type': 'pca', 'm': 11}
    # # MVG TIED # best gaussian model

    # print('R: MVG-Tied Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # scores_TiedGaussian, _ , _ = Evaluation.validate_final_model(TiedGaussian_Clf.TiedG(), DT, LT, DE, LE, preproc='znorm', dimred=dim_red, iprint=True)
    
    # ##################################
    # ### LINEAR LOGISTIC REGRESSION ###
    # ##################################
    
    # lbd = 10**-4 # best LLR model
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # scores_LLR, _ , _ = Evaluation.validate_final_model(LLR_Clf.LinearLogisticRegression(lbd, prior_weighted=True, prior=0.5),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red, iprint=True)
    
    
    # ##################
    # ### LINEAR SVM ###
    # ##################
    
    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 1} 
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # scores_LinearSVM, _ , _ = Evaluation.validate_final_model(SVM_Clf.SVM(hparams, None),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        iprint=True)
    
    
    # ######################
    # ### RBF KERNEL SVM ###
    # ######################
    
    # hparams = {'K': 1, 'eps': 0, 'gamma': 10**-1, 'C': 10, 'c': 0, 'd': 1} 
    # print('R: RBF K SVM\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # scores_RBFSVM, _ , _ = Evaluation.validate_final_model(SVM_Clf.SVM(hparams, kernel='RBF'),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        iprint=True)
    
    # ###########
    # ### GMM ###
    # ###########
    
    # nComponents = 4
    # cov = 'Tied' # best GMM model
    # print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    # scores_TiedGMM, _ , _ = Evaluation.validate_final_model(GMM_Clf.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red, iprint=True)
    
    # #####################################
    # # EXPERIMENTAL RESULTS PCA m = 11 ###
    # #####################################
    
    
    # ##########################
    # # GENERATIVE GAUSSIANS ###
    # ##########################

    # # MVG FULL

    # dim_red = {'type': 'pca', 'm': 11}
    # # MVG TIED # best gaussian model

    # print('R: MVG-Tied Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.validate_final_model(TiedGaussian_Clf.TiedG(), DT, LT,DE, LE, k=5, preproc='znorm', dimred=dim_red, iprint=True)
    
    # ##################################
    # ### LINEAR LOGISTIC REGRESSION ###
    # ##################################
    
    # lbd = 10**-4 # best LLR model
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # Evaluation.validate_final_model(LLR_Clf.LinearLogisticRegression(lbd, prior_weighted=True, prior=0.5),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red, iprint=True)
    
    
    
    # ##################
    # ### LINEAR SVM ###
    # ##################
    
    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 1} 
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.validate_final_model(SVM_Clf.SVM(hparams, None),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        iprint=True)
    
    
    # ######################
    # ### RBF KERNEL SVM ###
    # ######################
    
    # hparams = {'K': 1, 'eps': 0, 'gamma': 10**-1, 'C': 10, 'c': 0, 'd': 1} 
    # print('R: RBF K SVM\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # Evaluation.validate_final_model(SVM_Clf.SVM(hparams, kernel='RBF'),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        iprint=True)
    
    # ###########
    # ### GMM ###
    # ###########
    
    # nComponents = 4
    # cov = 'Full'
    # print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    # Evaluation.validate_final_model(GMM_Clf.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red, iprint=True)
    # cov = 'Tied'
    # print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    # Evaluation.validate_final_model(GMM_Clf.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red, iprint=True)


    # # ANALYSIS FOR SCORE IF THEY NEED CALIBRATION

    # #############################
    # ## SCORE CALIBRATION PLOT ###
    # #############################
    
    # fig1, axs1 = plt.subplots(2, 2)
    # fig1.set_figheight(5)
    # fig1.set_figwidth(13)
    # Evaluation.plot_Bayes_error_eval(ax=axs1[0, 0], title='Tied Covariance',
    #                                      model=TiedGaussian_Clf.TiedG(), preproc='znorm',
    #                                      dimred=None, DT=DT, LT=LT, DE=DE, LE=LE, calibrate_scores=False)
    # Evaluation.plot_Bayes_error_eval(ax=axs1[0, 1], title='Linear Logistic Regression',
    #                                      model=LLR_Clf.LinearLogisticRegression(lbd=10**-4, prior_weighted=True, prior=0.5), preproc='znorm',
    #                                      dimred=None, DT=DT, LT=LT, DE=DE, LE=LE, calibrate_scores=False)
    # Evaluation.plot_Bayes_error_eval(ax=axs1[1, 0], title='RBF Kernel SVM',
    #                                      model=SVM_Clf.SVM(hparams={'K': 1, 'C': 10, 'gamma': 10**-1}, kernel = 'RBF'), preproc='znorm',
    #                                      dimred=None, DT=DT, LT=LT, DE=DE, LE=LE, calibrate_scores=False)
    # Evaluation.plot_Bayes_error_eval(ax=axs1[1, 1], title='GMM',
    #                                      model=GMM_Clf.GMM(alpha=0.1, nComponents=4, psi=0.1, covType='Tied'), preproc='znorm',
    #                                      dimred=None, DT=DT, LT=LT, DE=DE, LE=LE, calibrate_scores=False)
    # fig1.tight_layout()
    # plt.show()

    # fig1, axs1 = plt.subplots(2, 2)
    # fig1.set_figheight(5)
    # fig1.set_figwidth(13)
    # Evaluation.plot_Bayes_error_eval(ax=axs1[0, 0], title='Tied Covariance',
    #                                      model=TiedGaussian_Clf.TiedG(), preproc='znorm',
    #                                      dimred=None, DT=DT, LT=LT,DE=DE, LE=LE, calibrate_scores=True)
    # Evaluation.plot_Bayes_error_eval(ax=axs1[0, 1], title='Linear Logistic Regression',
    #                                      model=LLR_Clf.LinearLogisticRegression(lbd=10**-4, prior_weighted=True, prior=0.5), preproc='znorm',
    #                                      dimred=None, DT=DT, LT=LT, DE=DE, LE=LE, calibrate_scores=True)
    # Evaluation.plot_Bayes_error_eval(ax=axs1[1, 0], title='RBF Kernel SVM',
    #                                      model=SVM_Clf.SVM(hparams={'K': 1, 'C': 10, 'gamma': 10**-1}, kernel = 'RBF'), preproc='znorm',
    #                                      dimred=None, DT=DT, LT=LT, DE=DE, LE=LE, calibrate_scores=True)
    # Evaluation.plot_Bayes_error_eval(ax=axs1[1, 1], title='GMM',
    #                                      model=GMM_Clf.GMM(alpha=0.1, nComponents=4, psi=0.1, covType='Tied'), preproc='znorm',
    #                                      dimred=None, DT=DT, LT=LT, DE=DE, LE=LE, calibrate_scores=True)
        
    # fig1.tight_layout()
    # plt.show()
    
    # # COMPARATION OF DCF/MINDCF FOR UNCALIBRATED/CALIBRATED SCORE
    
    # # UNCALIBRATED

    # ##########################
    # # GENERATIVE GAUSSIANS ###
    # ##########################

    # # MVG FULL

    # dim_red = None # {'type': 'pca', 'm': 11}
    # # # MVG TIED # best gaussian model

    # print('R: MVG-Tied Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # scores_TiedGaussian, _ , _ = Evaluation.validate_final_model(TiedGaussian_Clf.TiedG(), DT, LT, DE, LE, preproc='znorm', dimred=dim_red, iprint=True, prior=None, calibrated=False)
    
    # # ##################################
    # # ### LINEAR LOGISTIC REGRESSION ###
    # # ##################################
    
    # lbd = 10**-4 # best LLR model
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # scores_LLR, _ , _ = Evaluation.validate_final_model(LLR_Clf.LinearLogisticRegression(lbd, prior_weighted=True, prior=0.5),
    #                                         DT,
    #                                         LT,
    #                                         DE,
    #                                         LE,
    #                                         preproc='znorm',
    #                                         dimred=dim_red, iprint=True, prior=None, calibrated=False)
    
    
    # # ##################
    # # ### LINEAR SVM ###
    # # ##################
    
    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 1} 
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # scores_LinearSVM, _ , _ = Evaluation.validate_final_model(SVM_Clf.SVM(hparams, None),
    #                                         DT,
    #                                         LT,
    #                                         DE,
    #                                         LE,
    #                                         preproc='znorm',
    #                                         dimred=dim_red,
    #                                         iprint=True, prior=None, calibrated=False)
    
    # ######################
    # ### RBF KERNEL SVM ###
    # ######################
    
    # hparams = {'K': 1, 'eps': 0, 'gamma': 10**-1, 'C': 10, 'c': 0, 'd': 1} 
    # print('R: RBF K SVM\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # scores_RBFSVM, _ , _ = Evaluation.validate_final_model(SVM_Clf.SVM(hparams, kernel='RBF'),
    #                                          DT,
    #                                          LT,
    #                                          DE,
    #                                          LE,
    #                                          preproc='znorm',
    #                                          dimred=dim_red,
    #                                          iprint=True, prior=None, calibrated=False)
    
    # # # ###########
    # # # ### GMM ###
    # # # ###########
    
    # nComponents = 4
    # cov = 'Tied' # best GMM model
    # print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    # scores_TiedGMM, _ , _ = Evaluation.validate_final_model(GMM_Clf.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
    #                                          DT,
    #                                          LT,
    #                                          DE,
    #                                          LE,
    #                                          preproc='znorm',
    #                                          dimred=dim_red, iprint=True, prior=None, calibrated=False)
    
    # # CALIBRATED

    # ##########################
    # # GENERATIVE GAUSSIANS ###
    # ##########################

    # # MVG FULL

    # dim_red = None # {'type': 'pca', 'm': 11}
    # # # MVG TIED # best gaussian model

    # print('R: MVG-Tied Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # scores_TiedGaussian, _ , _ = Evaluation.validate_final_model(TiedGaussian_Clf.TiedG(), DT, LT, DE, LE, preproc='znorm', dimred=dim_red, iprint=True, prior=None, calibrated=True)
    
    # # ##################################
    # # ### LINEAR LOGISTIC REGRESSION ###
    # # ##################################
    
    # lbd = 10**-4 # best LLR model
    # print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # scores_LLR, _ , _ = Evaluation.validate_final_model(LLR_Clf.LinearLogisticRegression(lbd, prior_weighted=True, prior=0.5),
    #                                         DT,
    #                                         LT,
    #                                         DE,
    #                                         LE,
    #                                         preproc='znorm',
    #                                         dimred=dim_red, iprint=True, prior=None, calibrated=True)
    
    
    # # ##################
    # # ### LINEAR SVM ###
    # # ##################
    
    # hparams = {'K': 0, 'eps': 1, 'gamma': 1, 'C': 1} 
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # scores_LinearSVM, _ , _ = Evaluation.validate_final_model(SVM_Clf.SVM(hparams, None),
    #                                         DT,
    #                                         LT,
    #                                         DE,
    #                                         LE,
    #                                         preproc='znorm',
    #                                         dimred=dim_red,
    #                                         iprint=True, prior=None, calibrated=True)
    
    
    # # ######################
    # # ### RBF KERNEL SVM ###
    # # ######################
    
    # hparams = {'K': 1, 'eps': 0, 'gamma': 10**-1, 'C': 10, 'c': 0, 'd': 1} 
    # print('R: RBF K SVM\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # scores_RBFSVM, _ , _ = Evaluation.validate_final_model(SVM_Clf.SVM(hparams, kernel='RBF'),
    #                                         DT,
    #                                         LT,
    #                                         DE,
    #                                         LE,
    #                                         preproc='znorm',
    #                                         dimred=dim_red,
    #                                         iprint=True, prior=None, calibrated=True)
    
    # # ###########
    # # ### GMM ###
    # # ###########
    
    # nComponents = 4
    # cov = 'Tied' # best GMM model
    # print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    # scores_TiedGMM, _ , _ = Evaluation.validate_final_model(GMM_Clf.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
    #                                         DT,
    #                                         LT,
    #                                         DE,
    #                                         LE,
    #                                         preproc='znorm',
    #                                         dimred=dim_red, iprint=True, prior=None, calibrated=True)
    
    # #################
    # ## ROC CURVE   ##
    # #################
    
    
    # fig = plt.figure()
    # Evaluation.plotROC(plt, scores_TiedGaussian, LE)
    # Evaluation.plotROC(plt, scores_LLR, LE)
    # Evaluation.plotROC(plt, scores_LinearSVM, LE)
    # Evaluation.plotROC(plt, scores_RBFSVM, LE)
    # Evaluation.plotROC(plt, scores_TiedGMM, LE)
    # fig.legend(['Tied Gaussian', 'LLR', 'Linear SVM', 'RBF SVM' ,'Tied GMM(4 components)'])
    
    # plt.ylabel('TPR')
    # plt.xlabel('FPR')
    # plt.show()

    # # ANALYSIS OF SCORE FOR OTHER HYPERPARAMS

    # ##########################
    # # GENERATIVE GAUSSIANS ###
    # ##########################

    # # MVG FULL 
    # dim_red = None 
    # print('R: MVG-Full Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # scores_FullGaussian_noPCA, _ , _ = Evaluation.validate_final_model(MVG_Clf.MVG(), DT, LT, DE, LE, preproc='znorm', dimred=dim_red, iprint=True)

    # dim_red = {'type': 'pca', 'm': 11} 
    # print('R: MVG-Full Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # scores_FullGaussian_11PCA, _ , _ = Evaluation.validate_final_model(MVG_Clf.MVG(), DT, LT, DE, LE, preproc='znorm', dimred=dim_red, iprint=True)

    # # tied could not change anything so it won't be seen again

    # ##################################
    # ### LINEAR LOGISTIC REGRESSION ###
    # ##################################

    # # not prior weighted
    # dim_red = None

    # lbd = 10**-4
    # print('R: Linear Logistic Regression no pw\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # scores_LLR_nopw, _ , _ = Evaluation.validate_final_model(LLR_Clf.LinearLogisticRegression(lbd, prior_weighted=False),
    #                                         DT,
    #                                         LT,
    #                                         DE,
    #                                         LE,
    #                                         preproc='znorm',
    #                                         dimred=dim_red, iprint=True)

    # # Prior 0.1
    # print('R: Linear Logistic Regression pw-0.1 \nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # scores_LLR_pw01, _ , _ = Evaluation.validate_final_model(LLR_Clf.LinearLogisticRegression(lbd, prior_weighted=True, prior=0.1),
    #                                         DT,
    #                                         LT,
    #                                         DE,
    #                                         LE,
    #                                         preproc='znorm',
    #                                         dimred=dim_red, iprint=True)

    # # Prior 0.9
    # print('R: Linear Logistic Regression pw-0.9 \nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    # scores_LLR_pw09, _ , _ = Evaluation.validate_final_model(LLR_Clf.LinearLogisticRegression(lbd, prior_weighted=True, prior=0.9),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red, iprint=True)



    # ##################
    # ### LINEAR SVM ###
    # ##################
    
    # # different C=0.1

    # hparams = {'K': 0, 'eps': 1, 'C': 1} 
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # scores_LinearSVM, _ , _ = Evaluation.validate_final_model(SVM_Clf.SVM(hparams, None),
    #                                         DT,
    #                                         LT,
    #                                         DE,
    #                                         LE,
    #                                         preproc='znorm',
    #                                         dimred=dim_red,
    #                                         iprint=True)
    

    # # different C=1
    # hparams = {'K': 0, 'eps': 1, 'C': 1} 
    # print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # scores_LinearSVM, _ , _ = Evaluation.validate_final_model(SVM_Clf.SVM(hparams, None),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        iprint=True)
    
    # ######################
    # ### RBF KERNEL SVM ###
    # ######################
    

    # # gamma 0.001
    # hparams = {'K': 1, 'eps': 0, 'gamma': 10**-3, 'C': 10, 'c': 0, 'd': 1} 
    # print('R: RBF K SVM\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # scores_RBFSVM_g0001, _ , _ = Evaluation.validate_final_model(SVM_Clf.SVM(hparams, kernel='RBF'),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        iprint=True)


    # # gamma 1
    # hparams = {'K': 1, 'eps': 0, 'gamma': 1, 'C': 10, 'c': 0, 'd': 1} 
    # print('R: RBF K SVM\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    # scores_RBFSVM_g1, _ , _ = Evaluation.validate_final_model(SVM_Clf.SVM(hparams, kernel='RBF'),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red,
    #                                        iprint=True)


    # # GMM 
    # # try also the full with 4 components

    # nComponents = 4
    # cov = 'Full'
    # print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    # scores_GMMFull_4c, _ , _ = Evaluation.validate_final_model(GMM_Clf.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
    #                                         DT,
    #                                         LT,
    #                                         DE,
    #                                         LE,
    #                                         preproc='znorm',
    #                                         dimred=dim_red, iprint=True)

    # nComponents = 8
    # cov = 'Full'
    # print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    # scores_GMMFull_8c, _ , _ = Evaluation.validate_final_model(GMM_Clf.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red, iprint=True)



    # cov = 'Tied' # best GMM model
    # nComponents = 8
    # print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    # scores_TiedGMM_8c, _ , _ = Evaluation.validate_final_model(GMM_Clf.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
    #                                        DT,
    #                                        LT,
    #                                        DE,
    #                                        LE,
    #                                        preproc='znorm',
    #                                        dimred=dim_red, iprint=True)