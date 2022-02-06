

import numpy as np


def read_grid(model_name, n_feats):
    """
    load base parameter grid for clasifier
    """
    param_grid = master_grid[model_name]
    if model_name in ['RandomForestClassifier', 'DecisionTreeClassifier']:
        param_grid['max_features'] = list(np.arange(1,n_feats,5)) + ['auto','sqrt']
    
    return param_grid
    
    

master_grid = {
    'RandomForestClassifier' : { 'bootstrap': [True, False],
                                 'max_depth': list(np.arange(1,8,1)) + [None],
                                 'min_samples_leaf': list(np.arange(2,12,1)),
                                 'min_samples_split': list(np.arange(2,12,1)),
                                 # bc of bagging more estimators = higher perf theoretically so dont tune till last
                                 'n_estimators' : [500],
                                 'n_jobs' : [-1]
                                 
                                },
    'LogisticRegression'      : { 'penalty' : ['l1'],
                                  'C': list(np.arange(0.001,1.3,0.1)),
                                  'solver': ['saga'],
                                },
    'DecisionTreeClassifier'  : { 'max_depth' : list(np.arange(1,32,2)) + [None],
                                  'min_samples_split' : list(np.arange(0.02, 1.02, 0.02)),
                                  'min_samples_leafs' : list(np.arange(0.02, 1.02, 0.02)),
                                },
    'SVC'                     : { 'kernel': ['rbf','sigmoid','linear'], 
                                  'gamma': list(np.random.exponential(10, 100)),
                                  'C': list(np.random.exponential(10, 100)),
                                },     
    'GaussianNB'              : { 
                                
                                },
    'XGBoost'                 : { 'learning_rate': [0.01,0.001],
                                  'n_estimators': [500],
                                  'subsample': list(np.arange(0.1,0.9,0.05)),
                                  'max_depth': list(np.arange(1,13,1)) + [None],
                                  'lambda': [1],
                                  'gamma': list(np.arange(0.05,1,0.05)),
                                  'colsample_bytree': list(np.arange(0.1,0.9,0.10)),
                                  'objective':['binary:logistic'],
                                  'eval_metric':['logloss']
            
    }
}