#dont ever ever ever ever suppress all warnings...
import warnings
warnings.filterwarnings("ignore")
import logging
import os
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import itertools
# for saving results  
from data_science.nodes.param_grid import read_grid
import json
from sklearn.model_selection import train_test_split

class modelOpt():

    def __init__(self, X, y, model_list
                ,low_memory =False, log = True):
        """
        """
        self.X = X
        self.y = y
        self.model_list = self.validate_model_list(model_list)
        self.low_memory = low_memory
        if log:
            logging.basicConfig(level=logging.INFO)
            self.logging = logging
        # making results private so we cant break it accidentally
        self.__results = {}
        self.seed = np.random.randint(0,1000)
        self.problem_type = self.get_problem_type(y)
        self.logging.info('random seed: {}'.format(self.seed))
        self.results_dir = 'logs/results_{}'.format(datetime.datetime.now().strftime("%Y_%m_%d-%H%M"))
        self.trained_models = {}
        self.trained_scalers = {}
        return
        

    @staticmethod
    def get_problem_type(y):
        "infers classification or regression based on y"
        if pd.Series(y).nunique() <= 10:
            return 'clf'
        else:
            return 'reg'
            
    @staticmethod
    def validate_model_list(model_list):
        for model in model_list:
            if model not in ['RandomForestClassifier',
                             'LogisticRegression',
                             'DecisionTreeClassifier' , 
                             'SVC',
                             'GaussianNB',
                             'XGBoost'
                             ]:
                print(model,'error')
        return model_list
        
    @property
    def results(self):
        """
        getter for __results
        """
        return self.__results
    
    def set_up_results(self,model_name):
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        self.__results[model_name] = {}
        return
        
    def get_experiment_grid(self, model_name, max_iter = 100):
        """
        generate parameter grids for given models
        """
        param_grid = read_grid(model_name, self.X.shape[1])
        keys, values = zip(*param_grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        if max_iter >= len(experiments):
            self.logging.info('use grid search, {}'.format(len(experiments)))
            iters = len(experiments)-1
        else:
            iters = max_iter
        np.random.seed(self.seed)
        sample_array = list(np.random.choice(range(len(experiments)), iters, replace=False))
        return [experiments[xi] for xi in sample_array]
    
    def train_model(self, model_name, cv=StratifiedShuffleSplit(5), max_iter=100, scaling = MinMaxScaler()):
        """
        trains the model on given X, y using cv supplied. Standard cv is sss with 5 splits.
        https://stats.stackexchange.com/questions/160479/practical-hyperparameter-optimization-random-vs-grid-search
        95% chance to achieve a result theoretically within 5% of true max need 60 trials
        95% chance to achieve a result theoretically within 1% of true max need 300 trials
        Comparatively an exhaustive search needs n trials.
        not using sklearn random search bc we want to have control over the results + able to save them down periodically+
        scale each fold independantly to prevent leakage.
        """
        X = self.X.copy()
        y = self.y.copy()
        self.set_up_results(model_name)
        self.logging.info("Training model: {} for {} iterations".format(model_name, max_iter))
        # load experiments
        experiments = self.get_experiment_grid(model_name,max_iter)
        for num, xpmnt in enumerate(experiments):
            self.logging.info("running experiment {} with parameters: {}".format(num, xpmnt))
            cv_scores={}
            # set up crossvalidation
            for train_index, test_index in cv.split(X,y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                # scale data here to not contaminate the cv sets
                if scaling:
                    X_train, scl = self.build_scaler(X_train)
                    X_test = scl.transform(X_test)
                # fit model on training idxs
                mod = self.load_model(model_name,xpmnt)
                mod.fit(X_train,y_train)
                #update cv results
                cv_scores = self.evaluate_performance(mod, X_test, y_test, cv_scores)
            #update master results
            self.update_results(model_name,num,cv_scores,xpmnt)
            
            if num % 5 == 0:
                # print(self.__results)
                self.save_to_disk(model_name,num)
        # save final results
        self.save_to_disk(model_name,num)   
        return
        
    @staticmethod
    def load_model(model_name, params):
        """
        sets up model for experiment
        """
        model_dict={
                    'RandomForestClassifier':RandomForestClassifier,
                    'LogisticRegression': LogisticRegression,
                    'DecisionTreeClassifier' : DecisionTreeClassifier,
                    'SVC' : SVC,
                    'GaussianNB':GaussianNB,
                    'XGBoost' : XGBClassifier
                    }
        
        return model_dict[model_name](**params)
        
    
    def save_to_disk(self, model_name, iter):
        """
        save dicionary of results to disk. 
        used to save on RAM
        """
        # print(type(self.__results))
        
        with open('{}/{}_{}.txt'.format(self.results_dir,model_name,iter), 'w') as file:
            file.write(json.dumps(self.__results))
        if self.low_memory:
            self.set_up_results(model_name)
        return
    
    def evaluate_performance(self,mod,X_test,y_test,cv_scores):
        """
        """
        prd=mod.predict_proba(X_test)[:,1]
        
        if self.problem_type == 'clf':
            if cv_scores:
                cv_scores['rocauc'].append(roc_auc_score(y_test,prd))
                cv_scores['prauc'].append(average_precision_score(y_test,prd))
            else:
                cv_scores['rocauc'] = [roc_auc_score(y_test,prd)]
                cv_scores['prauc']  = [average_precision_score(y_test,prd)]
        
        else:
            raise('NO REGRESSION YET')
        return cv_scores
    
    def update_results(self,model_name,iter,scores,params):
        """
        update master results dictionary with iteration results
        """
        sc = {}
        
        for key in scores.keys():
            sc [key]={
                     'raw':scores[key],
                     'mean':np.mean(scores[key]),
                     'stddev': np.std(scores[key]),
                    }      
        self.__results[model_name][iter] = sc
        for key in params:
            params[key] = str(params[key])
        self.__results[model_name][iter]['params']=params
        
        self.logging.info('results for iteration: {}'.format(iter))
        for key in scores.keys():
            self.logging.info('mean {}: {}'.format(key, self.__results[model_name][iter][key]['mean']))
            self.logging.info('stddev {}: {}'.format(key, self.__results[model_name][iter][key]['stddev']))
            
        return 
        
    @staticmethod
    def build_scaler(X, scaler=MinMaxScaler()):
        """
        returns scaled train df and scaler object to save to disk.
        """
        scl = scaler
        scl.fit(X)
        X=scl.transform(X)
        return X, scl
        
    def optimize(self):
        """
        CBA to build your own pipe? no problem. just run this method it will run the opter with the defualt params
        """
        for model_name in self.model_list:
            self.train_model(model_name)
            
    def handle_low_memory():
        """
        potentially wanting to save down x and load on the fly. Slow but would prevent memory issues.
        """
        if self.low_memory:
            self.X.to_csv('x_temp.csv',index= False)
        return 
    
    def train_model_full(self, model_name, params):
        """
        train a model without cv
        """
        X = self.X.copy()
        y = self.y.copy()
        clf = self.load_model(model_name, params)
        X, scl = self.build_scaler(X)
        clf.fit(X,y)
        self.trained_models[model_name] = clf
        self.trained_scalers[model_name] = scl
        return
    
    def score_new_data_set(self, test_x, model_name):
        """
        scores new data
        """
        clf = self.trained_models[model_name]
        scl = self.trained_scalers[model_name]
        test_x = scl.transform(test_x)
        return clf.predict_proba(test_x)
        

def prepare_ml(df_master, config: dict):
    """

    :param df_master:
    :param config:
    :return:
    """
    training_config = config['modelling']
    FEATURES = list(set(training_config['features']))
    targets = training_config['target']
    logging.info('splitting on id')
    train_size = training_config.get('train_size',0.7)
    a = train_test_split(df_master[config['id']].unique(), random_state=1993, train_size=train_size)[0]
    train_index = df_master.index[
        df_master[config['id']].isin(a)
    ].to_numpy()
    test_index = df_master.index[
        ~df_master[config['id']].isin(a)
    ].to_numpy()

    df_master = df_master[FEATURES+targets]
    df_master = df_master.rename(columns={targets[0]:'target'})
    
    df_master['target'] = df_master['target'].astype(int)

    num_cols = list(df_master.columns[df_master.dtypes != object])

    logging.info(f'train index is {len(train_index)} in length')
    logging.info(f'test index is {len(test_index)} in length')
    # df_master = pd.concat([df_master,bin_master], axis =1)

    df_master_train = df_master.loc[train_index]
    df_master_test = df_master.loc[test_index]
    
    return df_master_train, df_master_test
    


def train_models(df_master_train,ml_params):
    X= df_master_train.copy() 
    y = X.pop('target')
    X=X.fillna(0)
    ml=modelOpt(X,y,ml_params['models'])
    ml.optimize()
    return ml
    
def get_best_params(ml):
    res = ml.results
    best_params={}
    for k in res.keys():
        best_params[k]={}
        best_params[k]['best_score'] = 0.5
        best_params[k]['best_iter']= 0
        for i in res[k].keys():
            if res[k][i]['rocauc']['mean'] > best_params[k]['best_score']:
                best_params[k]['best_score'] = res[k][i]['rocauc']['mean']
                best_params[k]['best_iter'] = i
                best_params[k]['params'] = res[k][i]['params']
    return best_params

def get_best_models(ml, best_params, ml_params):
    """
    """
    for model_name in ml_params['models']:
        ml.train_model_full(model_name,best_params[model_name]['params'])
        
    return ml


    