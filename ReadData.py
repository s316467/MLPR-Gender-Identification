import numpy as np
import Constants as CONST

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1,v.size))

def read_data_training(path):
    DT = np.zeros(shape=(CONST.NUM_FEATURES, CONST.NUM_TSAMPLES), dtype='float32')
    LT = np.zeros(CONST.NUM_TSAMPLES, dtype='int32')
    with open(path, 'r') as file:
        i = 0
        for line in file:
            features_list = line.split(',')
            features = np.array(features_list[0:CONST.NUM_FEATURES], dtype='float32').reshape(-1, 1)
            label = int(features_list[-1])
            DT[:, i:i + 1] = features
            LT[i] = label
            i += 1
    return DT, LT

def read_data_evaluation(path):
    DE = np.zeros(shape=(CONST.NUM_FEATURES, CONST.NUM_ESAMPLES), dtype='float32')
    LE = np.zeros(CONST.NUM_ESAMPLES, dtype='int32')
    with open(path, 'r') as file:
        i = 0
        for line in file:
            features_list = line.split(',')
            features = np.array(features_list[0:CONST.NUM_FEATURES], dtype='float32').reshape(-1, 1)
            label = int(features_list[-1])
            DE[:, i:i + 1] = features
            LE[i] = label
            i += 1
    return DE, LE
