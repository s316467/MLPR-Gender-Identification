import numpy as np
import matplotlib.pyplot as plt
import Dim_reduction as DimRed
import FeatureAnalysis as FA
import LLR_Clf as LLR

    
def accuracy(conf_matrix):
    return (conf_matrix[0, 0] + conf_matrix[1, 1]) / (conf_matrix[0, 0] + conf_matrix[1, 1] + conf_matrix[1, 0] + conf_matrix[0, 1])


def error_rate(predicted_labels, actual_labels):
    return 1 - np.sum([predicted_labels == actual_labels]) / len(predicted_labels)


def confusion_matrix(predicted_labels, actual_labels):
    conf_matrix = np.zeros(shape=(2, 2))
    i = 0
    for i in range(2):
        for j in range(2):
            conf_matrix[i, j] = np.sum((predicted_labels == i) & (actual_labels == j))
    return conf_matrix


def optimalBayes_confusion_matrix(scores, actual_labels, pi1, Cfn, Cfp):
    t = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
    predicted_labels = np.array(scores > t, dtype='int32')
    return confusion_matrix(predicted_labels, actual_labels)


def DCFu(scores, actual_labels, pi1, Cfn, Cfp):
    M = optimalBayes_confusion_matrix(scores, actual_labels, pi1, Cfn, Cfp)
    fnr = M[0, 1] / (M[0, 1] + M[1, 1])
    fpr = M[1, 0] / (M[0, 0] + M[1, 0])
    return pi1 * Cfn * fnr + (1 - pi1) * Cfp * fpr


def DCF(scores, actual_labels, pi1, Cfn, Cfp):
    Bdummy_DCF = np.minimum(pi1 * Cfn, (1 - pi1) * Cfp)
    return DCFu(scores, actual_labels, pi1, Cfn, Cfp) / Bdummy_DCF


def minDCF(scores, actual_labels, pi1, Cfn, Cfp):
    scores_sort = np.sort(scores)
    normDCFs = []
    for t in scores_sort:
        predicted_labels = np.where(scores > t + 0.000001, 1, 0)
        M = confusion_matrix(predicted_labels, actual_labels)
        fnr = M[0, 1] / (M[0, 1] + M[1, 1])
        fpr = M[1, 0] / (M[0, 0] + M[1, 0])
        dcf = pi1 * Cfn * fnr + (1 - pi1) * Cfp * fpr
        Bdummy_DCF = np.minimum(pi1 * Cfn, (1 - pi1) * Cfp)
        dcf_norm = dcf / Bdummy_DCF
        normDCFs.append(dcf_norm)
    return min(normDCFs)

def plotROC(plt=plt, llrs=None, LE=None):
    
    TPR = []
    FPR = []
    llrs_sort = np.sort(llrs)
    for i in llrs_sort:
        predicted_labels = np.where(llrs > i + 0.000001, 1, 0)
        conf_matrix = confusion_matrix(predicted_labels, LE)
        TPR.append(conf_matrix[1, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1]))
        FPR.append(conf_matrix[1, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0]))
    plt.plot(np.array(FPR), np.array(TPR))

def plot_Bayes_error(ax=None, title=None, model=None, preproc='raw', dimred=None, DT=None, LT=None, calibrate_scores=False):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    effPrior = 1 / (1 + np.exp(-effPriorLogOdds))
    mindcf = []
    dcf = []
    for prior in effPrior:
        DCF, minDCF = kfold_cross_validation(model, DT, LT, k=5, preproc=preproc, dimred=dimred, prior=prior, calibrated=calibrate_scores)
        mindcf.append(minDCF)
        dcf.append(DCF)
    ax.plot(effPriorLogOdds, dcf, label='DCF', color='red')
    ax.plot(effPriorLogOdds, mindcf, label='min DCF', color='blue', linestyle='dashed')
    ax.set_xlim([-3, 3])
    ax.legend(['DCF', 'min DCF'])
    ax.set_xlabel("prior log-odds")
    ax.set_title(title)


def plot_Bayes_error_eval(ax=None, title=None, model=None, preproc='raw', dimred=None, DT=None, LT=None, DE=None, LE=None,calibrate_scores=False):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    effPrior = 1 / (1 + np.exp(-effPriorLogOdds))
    mindcf = []
    dcf = []
    for prior in effPrior:
        DCF, minDCF = validate_final_model(model, DT, LT, DE, LE, preproc=preproc, dimred=dimred, iprint=True, prior=prior, calibrated=calibrate_scores)
        mindcf.append(minDCF)
        dcf.append(DCF)
    ax.plot(effPriorLogOdds, dcf, label='DCF', color='red')
    ax.plot(effPriorLogOdds, mindcf, label='min DCF', color='blue', linestyle='dashed')
    ax.set_xlim([-3, 3])
    ax.legend(['DCF', 'min DCF'])
    ax.set_xlabel("prior log-odds")
    ax.set_title(title)


def kfold_cross_validation(model=None, D=None, L=None, k=5, preproc='raw', dimred=None, iprint=True, prior=None, calibrated=False):
    Nsamples = D.shape[1]
    np.random.seed(0)
    idx = np.random.permutation(Nsamples)
    folds = np.array_split(idx, k)
    k_scores = []
    k_labels = []
    if dimred is not None:
        dimred_type = dimred['type']
        dimred_m = dimred['m']
    else:
        dimred_type = None

    for i in range(k):
        fold_test = folds[i]
        Dtest = D[:, fold_test]
        Ltest = L[fold_test]
        folds_train = []
        for j in range(k):
            if j != i:
                folds_train.append(folds[j])
        Dtrain = D[:, np.array(folds_train).flat]
        Ltrain = L[np.array(folds_train).flat]

        if preproc == 'gau':
            Dtrain_normalized = FA.gaussianized_features_training(Dtrain)
            Dtest_normalized = FA.gaussianized_features_evaluation(Dtest, Dtrain)
        elif preproc == 'znorm':
            Dtrain_normalized = FA.znormalized_features_training(Dtrain)
            Dtest_normalized = FA.znormalized_features_evaluation(Dtest, Dtrain)
        elif preproc == 'zg':
            Dtrain_z = FA.znormalized_features_training(Dtrain)
            Dtrain_normalized = FA.gaussianized_features_training(Dtrain_z)
            Dtest_z = FA.znormalized_features_evaluation(Dtest, Dtrain)
            Dtest_normalized = FA.gaussianized_features_evaluation(Dtest_z, Dtrain_normalized)
        else:
            Dtrain_normalized = Dtrain
            Dtest_normalized = Dtest

        if dimred_type == 'pca':
            P = DimRed.computeProjectionMatrix(Dtrain_normalized, dimred_m)
            Dtrain_normalized_reduced = DimRed.PCA(Dtrain_normalized, P)
            Dtest_normalized_reduced = DimRed.PCA(Dtest_normalized, P)
        else:
            Dtrain_normalized_reduced = Dtrain_normalized
            Dtest_normalized_reduced = Dtest_normalized

        k_scores.append(model.train(Dtrain_normalized_reduced, Ltrain).predict(Dtest_normalized_reduced, labels=False))
        k_labels.append(Ltest)

    k_scores = np.hstack(k_scores)
    k_labels = np.hstack(k_labels)

    if calibrated is True:
        s = k_scores.reshape(1, -1)
        p = 0.5
        lr = LLR.LinearLogisticRegression(lbd=10**-6, prior_weighted=True, prior=p)
        alpha, betafirst = lr.train(s, k_labels).get_model_parameters()
        k_scores_cal = (alpha*s + betafirst - np.log(p/(1-p))).reshape(Nsamples,)
    else:
        k_scores_cal = k_scores

    dcf = []
    min_dcf = []
    priors = [0.1, 0.5, 0.9]
    if prior is None:
        for p in priors:
            dcf.append(DCF(k_scores_cal, k_labels, p, 1, 1))
            min_dcf.append(minDCF(k_scores_cal, k_labels, p, 1, 1))
    else:
        return DCF(k_scores_cal, k_labels, prior, 1, 1), minDCF(k_scores_cal, k_labels, prior, 1, 1)

    if iprint:
        for i in range(len(priors)):
            print('pi=[%.1f] DCF = %.3f - minDCF = %.3f' % (priors[i], dcf[i], min_dcf[i]))
    return dcf, min_dcf


def validate_final_model(model=None, DT=None, LT=None, DE=None, LE=None, preproc='raw', dimred=None, iprint=True, prior=None, calibrated=False):
    
    if dimred is not None:
        dimred_type = dimred['type']
        m = dimred['m']

    if preproc == 'gau':
        DTnorm = FA.gaussianized_features_training(DT)
        DEnorm = FA.gaussianized_features_evaluation(DE, DT)
    elif preproc == 'znorm':
        DTnorm = FA.znormalized_features_training(DT)
        DEnorm = FA.znormalized_features_evaluation(DE, DT)
    elif preproc == 'zg':
        DTz = FA.znormalized_features_training(DT)
        DTnorm = FA.gaussianized_features_training(DTz)
        DEz = FA.znormalized_features_evaluation(DE, DT)
        DEnorm = FA.gaussianized_features_evaluation(DEz, DTnorm)
    else:
        DTnorm = DT
        DEnorm = DE

    if dimred is not None and dimred_type == 'pca':
        P = DimRed.computeProjectionMatrix(DTnorm, m)
        DTnorm_red = DimRed.PCA(DTnorm, P)
        DEnorm_red = DimRed.PCA(DEnorm, P)
    else:
        DTnorm_red = DTnorm
        DEnorm_red = DEnorm

    scores = model.train(DTnorm_red, LT).predict(DEnorm_red, labels=False)

    if calibrated is True:
        s = scores.reshape(1, -1)
        p = 0.5
        lr = LLR.LinearLogisticRegression(lbd=10**-6, prior_weighted=True, prior=p)
        alpha, betafirst = lr.train(s, LE).get_model_parameters()
        scores_cal = (alpha*s + betafirst - np.log(p/(1-p))).reshape(DE.shape[1],)
    else:
        scores_cal = scores

    dcf = []
    min_dcf = []
    priors = [0.1, 0.5, 0.9]
    if prior is None:
        for p in priors:
            dcf.append(DCF(scores_cal, LE, p, 1, 1))
            min_dcf.append(minDCF(scores_cal, LE, p, 1, 1))
    else:
        return DCF(scores_cal, LE, prior, 1, 1), minDCF(scores_cal, LE, prior, 1, 1)

    if iprint:
        for i in range(len(priors)):
            print('pi=[%.1f] DCF = %.3f - minDCF = %.3f' % (priors[i], dcf[i], min_dcf[i]))
    return scores_cal, dcf, min_dcf


def plot_lambda_minDCF_LLR(lambdas, mindcf_LLR_noPCA, mindcf_LLR_11PCA, mindcf_LLR_10PCA):
    
    fig1, axs1 = plt.subplots(1, 3)
    fig1.set_figheight(5)
    fig1.set_figwidth(13)
    colors = ["red", "blue", "green"]
    
    axs1[0].set_title('no PCA')
    axs1[0].plot(lambdas, mindcf_LLR_noPCA[0], label="π = 0.1", color=colors[0])
    axs1[0].plot(lambdas, mindcf_LLR_noPCA[1], label="π = 0.5", color=colors[1])
    axs1[0].plot(lambdas, mindcf_LLR_noPCA[2], label="π = 0.9", color=colors[2])
        
    axs1[0].set_xscale('log')
    axs1[0].set_xticks([1.E-6, 1.E-4, 1.E-2, 1, 100, 10000, 100000])
    axs1[0].set_xlabel("λ")
    axs1[0].set_ylim([0, 1])
    
    axs1[1].set_title('PCA m=11')
    axs1[1].plot(lambdas, mindcf_LLR_11PCA[0], label="π = 0.1", color=colors[0])
    axs1[1].plot(lambdas, mindcf_LLR_11PCA[1], label="π = 0.5", color=colors[1])
    axs1[1].plot(lambdas, mindcf_LLR_11PCA[2], label="π = 0.9", color=colors[2])
        
    axs1[1].set_xscale('log')
    axs1[1].set_xticks([1.E-6, 1.E-4, 1.E-2, 1, 100, 10000, 100000])
    axs1[1].set_xlabel("λ")
    axs1[1].set_ylim([0, 1])

    axs1[2].set_title('PCA m=10')
    axs1[2].plot(lambdas, mindcf_LLR_10PCA[0], label="π = 0.1", color=colors[0])
    axs1[2].plot(lambdas, mindcf_LLR_10PCA[1], label="π = 0.5", color=colors[1])
    axs1[2].plot(lambdas, mindcf_LLR_10PCA[2], label="π = 0.9", color=colors[2])
        
    axs1[2].set_xscale('log')
    axs1[2].set_xticks([1.E-6, 1.E-4, 1.E-2, 1, 100, 10000, 100000])
    axs1[2].set_xlabel("λ")
    axs1[2].set_ylim([0, 1])


    
    axs1[0].set_ylabel("minDCF")
    fig1.legend(['π = 0.1', 'π = 0.5', 'π = 0.9'], loc='lower right')
    plt.show()


def plot_lambda_minDCF_QLR(lambdas, mindcf_LLR_noPCA, mindcf_LLR_11PCA, mindcf_LLR_10PCA):
    fig1, axs1 = plt.subplots(1, 3)
    fig1.set_figheight(5)
    fig1.set_figwidth(13)
    colors = ["red", "blue", "green"]
    
    axs1[0].set_title('no PCA')
    axs1[0].plot(lambdas, mindcf_LLR_noPCA[0], label="π = 0.1", color=colors[0])
    axs1[0].plot(lambdas, mindcf_LLR_noPCA[1], label="π = 0.5", color=colors[1])
    axs1[0].plot(lambdas, mindcf_LLR_noPCA[2], label="π = 0.9", color=colors[2])
        
    axs1[0].set_xscale('log')
    axs1[0].set_xticks([1.E-6, 1.E-4, 1.E-2, 1, 100, 10000, 100000])
    axs1[0].set_xlabel("λ")
    axs1[0].set_ylim([0, 1])
    
    axs1[1].set_title('PCA m=11')
    axs1[1].plot(lambdas, mindcf_LLR_11PCA[0], label="π = 0.1", color=colors[0])
    axs1[1].plot(lambdas, mindcf_LLR_11PCA[1], label="π = 0.5", color=colors[1])
    axs1[1].plot(lambdas, mindcf_LLR_11PCA[2], label="π = 0.9", color=colors[2])
        
    axs1[1].set_xscale('log')
    axs1[1].set_xticks([1.E-6, 1.E-4, 1.E-2, 1, 100, 10000, 100000])
    axs1[1].set_xlabel("λ")
    axs1[1].set_ylim([0, 1])

    axs1[2].set_title('PCA m=10')
    axs1[2].plot(lambdas, mindcf_LLR_10PCA[0], label="π = 0.1", color=colors[0])
    axs1[2].plot(lambdas, mindcf_LLR_10PCA[1], label="π = 0.5", color=colors[1])
    axs1[2].plot(lambdas, mindcf_LLR_10PCA[2], label="π = 0.9", color=colors[2])
        
    axs1[2].set_xscale('log')
    axs1[2].set_xticks([1.E-6, 1.E-4, 1.E-2, 1, 100, 10000, 100000])
    axs1[2].set_xlabel("λ")
    axs1[2].set_ylim([0, 1])


    
    axs1[0].set_ylabel("minDCF")
    fig1.legend(['π = 0.1', 'π = 0.5', 'π = 0.9'], loc='lower right')
    plt.show()


def plot_lambda_minDCF_LinearSVM(C, mindcf_linearSVM_noPCA, mindcf_linearSVM_11PCA, mindcf_linearSVM_10PCA):
    
    fig1, axs1 = plt.subplots(1, 3)
    fig1.set_figheight(5)
    fig1.set_figwidth(13)
    colors = ["red", "blue", "green"]

    axs1[0].set_title('no PCA')
    axs1[0].plot(C, mindcf_linearSVM_noPCA[0], label="π = 0.1", color=colors[0])
    axs1[0].plot(C, mindcf_linearSVM_noPCA[1], label="π = 0.5", color=colors[1])
    axs1[0].plot(C, mindcf_linearSVM_noPCA[2], label="π = 0.9", color=colors[2])
    axs1[0].set_xscale('log')
    axs1[0].set_xticks(C)
    axs1[0].set_xlabel("C")
    axs1[0].set_ylim([0, 1])

    axs1[1].set_title('PCA m=11')
    axs1[1].plot(C, mindcf_linearSVM_11PCA[0], label="π = 0.1", color=colors[0])
    axs1[1].plot(C, mindcf_linearSVM_11PCA[1], label="π = 0.5", color=colors[1])
    axs1[1].plot(C, mindcf_linearSVM_11PCA[2], label="π = 0.9", color=colors[2])
    axs1[1].set_xscale('log')
    axs1[1].set_xticks(C)
    axs1[1].set_xlabel("C")
    axs1[1].set_ylim([0, 1])

    axs1[2].set_title('PCA m=10')
    axs1[2].plot(C, mindcf_linearSVM_10PCA[0], label="π = 0.1", color=colors[0])
    axs1[2].plot(C, mindcf_linearSVM_10PCA[1], label="π = 0.5", color=colors[1])
    axs1[2].plot(C, mindcf_linearSVM_10PCA[2], label="π = 0.9", color=colors[2])
    axs1[2].set_xscale('log')
    axs1[2].set_xticks(C)
    axs1[2].set_xlabel("C")
    axs1[2].set_ylim([0, 1])
    
    axs1[0].set_ylabel("minDCF")
    fig1.legend(['π = 0.1', 'π = 0.5', 'π = 0.9'], loc='lower right')
    plt.show()


def plot_lambda_minDCF_RBFSVM(C, mindcf_RBFSVM_001g, mindcf_RBFSVM_01g, mindcf_RBFSVM_1g):

    fig1, axs1 = plt.subplots(1, 3)
    fig1.set_figheight(5)
    fig1.set_figwidth(13)
    colors = ["red", "blue", "green"]
    axs1[0].set_title('γ = 0.01')
    axs1[0].plot(C, mindcf_RBFSVM_001g[0], label="π = 0.1", color=colors[0])
    axs1[0].plot(C, mindcf_RBFSVM_001g[1], label="π = 0.5", color=colors[1])
    axs1[0].plot(C, mindcf_RBFSVM_001g[2], label="π = 0.9", color=colors[2])
    axs1[0].set_xscale('log')
    axs1[0].set_xticks(C)
    axs1[0].set_xlabel("C")
    axs1[0].set_ylim([0, 1.2])
    axs1[0].set_ylabel("minDCF")

    axs1[1].set_title('γ = 0.1')
    axs1[1].plot(C, mindcf_RBFSVM_01g[0], label="π = 0.1", color=colors[0])
    axs1[1].plot(C, mindcf_RBFSVM_01g[1], label="π = 0.5", color=colors[1])
    axs1[1].plot(C, mindcf_RBFSVM_01g[2], label="π = 0.9", color=colors[2])
    axs1[1].set_xscale('log')
    axs1[1].set_xticks(C)
    axs1[1].set_xlabel("C")
    axs1[1].set_ylim([0, 1.2])
    axs1[1].set_ylabel("minDCF")
    
    axs1[2].set_title('γ = 1')
    axs1[2].plot(C, mindcf_RBFSVM_1g[0], label="π = 0.1", color=colors[0])
    axs1[2].plot(C, mindcf_RBFSVM_1g[1], label="π = 0.5", color=colors[1])
    axs1[2].plot(C, mindcf_RBFSVM_1g[2], label="π = 0.9", color=colors[2])
    axs1[2].set_xscale('log')
    axs1[2].set_xticks(C)
    axs1[2].set_xlabel("C")
    axs1[2].set_ylim([0, 1.2])
    axs1[2].set_ylabel("minDCF")
    
    

    fig1.legend(['π = 0.1', 'π = 0.5', 'π = 0.9'], loc='lower right')
    plt.show()


def plot_gaussian_models(mindcf_MVG, mindcf_Tied, mindcf_Naive):
    
    fig1, axs1 = plt.subplots(1, 3)
    x = [0.1, 0.5, 0.9]
    
    axs1[0].plot(x, mindcf_MVG[0], '--o', label='Full')
    axs1[0].plot(x, mindcf_Tied[0], '--o', label='Tied')
    axs1[0].plot(x, mindcf_Naive[0], '--o', label='Naive')
    axs1[0].set_ylim(0, 1)
    axs1[0].set_title('no PCA')
    axs1[0].set_xticks([0.1, 0.5, 0.9])
    axs1[0].set_xlabel("π")

    axs1[1].plot(x, mindcf_MVG[1], '--o', label='Full')
    axs1[1].plot(x, mindcf_Tied[1], '--o', label='Tied')
    axs1[1].plot(x, mindcf_Naive[1], '--o', label='Naive')
    axs1[1].set_ylim(0, 1)
    axs1[1].set_title('PCA(m=11)')
    axs1[1].set_xticks([0.1, 0.5, 0.9])
    axs1[1].set_xlabel("π")
    
    axs1[2].plot(x, mindcf_MVG[2], '--o', label='Full')
    axs1[2].plot(x,  mindcf_Tied[2], '--o', label='Tied')
    axs1[2].plot(x, mindcf_Naive[2], '--o', label='Naive')
    axs1[2].set_ylim(0, 1)
    axs1[2].set_title('PCA(m=10)')
    axs1[2].set_xticks([0.1, 0.5, 0.9])
    axs1[2].set_xlabel("π")

    fig1.legend(['Full', 'Tied', 'Naive'], loc='lower right')
    fig1.tight_layout()
    plt.show()


def plot_histogramGMM(comp, mindcf_GMM_noPCA_05p):
    fig1, axs1 = plt.subplots(1, 3, constrained_layout = True)
    fig1.set_figheight(5)
    fig1.set_figwidth(13)
    colors = ["red", "blue", "green"]
    
    ind = np.arange(len(comp))
    axs1[0].bar(x=ind, height=mindcf_GMM_noPCA_05p[0][0], width=0.25, alpha=0.6, color=colors[0])
    axs1[0].bar(x=ind+0.25, height=mindcf_GMM_noPCA_05p[0][1], width=0.25, alpha=0.6, color=colors[1])
    axs1[0].set_xticks(ind+0.25, comp)
    axs1[0].set_xlabel("GMM components")
    axs1[0].set_title('Full')
    axs1[0].set_ylim([0, 0.6])    
    axs1[0].set_ylabel("minDCF")

    axs1[1].bar(x=ind, height=mindcf_GMM_noPCA_05p[1][0], width=0.25, alpha=0.6, color=colors[0])
    axs1[1].bar(x=ind+0.25, height=mindcf_GMM_noPCA_05p[1][1], width=0.25, alpha=0.6, color=colors[1])
    axs1[1].set_xticks(ind+0.25, comp)
    axs1[1].set_xlabel("GMM components")
    axs1[1].set_title('Tied')
    axs1[1].set_ylim([0, 0.6])    
    axs1[1].set_ylabel("minDCF")

    axs1[2].bar(x=ind, height=mindcf_GMM_noPCA_05p[2][0], width=0.25, alpha=0.6, color=colors[0])
    axs1[2].bar(x=ind+0.25, height=mindcf_GMM_noPCA_05p[2][1], width=0.25, alpha=0.6, color=colors[1])
    axs1[2].set_xticks(ind+0.25, comp)
    axs1[2].set_xlabel("GMM components")
    axs1[2].set_title('Diag')
    axs1[2].set_ylim([0, 0.6])    
    axs1[2].set_ylabel("minDCF")



    fig1.legend(['minDCF(π = 0.5) - Z-Norm Features', 'minDCF(π = 0.5) - Gaussianization'], loc='lower right')
    plt.show()

