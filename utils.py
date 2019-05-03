import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fowlkes_mallows_score


def load(path):
    df = np.loadtxt(path)
    df = pd.DataFrame(df, columns=['brick_id', 'SX', 'SY', 'SZ', 'TX', 'TY']) 
    return df



def scorer(labels_true, labels_pred):
    groups, labels_true = labels_true.T
    _, labels_pred = labels_pred.T
    if groups is None:
        return fowlkes_mallows_score(labels_true=labels_true, labels_pred=labels_pred)
    fowlkes_mallows = 0.
    for group in np.unique(groups):
        fowlkes_mallows += fowlkes_mallows_score(labels_true=labels_true[groups==group], 
                                                 labels_pred=labels_pred[groups==group])
    return fowlkes_mallows / len(np.unique(groups))







