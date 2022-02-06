import sklearn.metrics as metrics
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from data_science.nodes.logistic import build_dummies, fill_in_test


def roc_chart(fpr_train, tpr_train, roc_train, fpr_test, tpr_test, roc_test):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_train, tpr_train, '#f6b26b', label = 'AUC Train = %0.2f' % roc_train)
    plt.plot(fpr_test, tpr_test, '#fac185', label = 'AUC Test = %0.2f' % roc_test)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'p--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return plt

def roc_data(df, model, model_variables):

    df_t = build_dummies(df)
    df_dummies = fill_in_test(df_t, model_variables)
    probs = model.predict_proba(df_dummies[model_variables])
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(df['target'], preds)
    roc_auc= metrics.auc(fpr, tpr)
    print(roc_auc)
    return fpr, tpr, roc_auc
