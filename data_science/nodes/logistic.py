import os
import warnings

warnings.filterwarnings("ignore")
import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV



def prepare_logistic(df_master, config: dict):
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
    print(df_master['target'].head())
    df_master['target'] = df_master['target'].astype(int)
    # print(df_master['target'].head())

    remove_large_string = df_master.columns[df_master.nunique() > 50] & df_master.columns[df_master.dtypes == object]
    df_master = df_master.drop(remove_large_string, axis=1)

    num_cols = list(df_master.columns[df_master.dtypes != object])

    logging.info(f'train index is {len(train_index)} in length')
    logging.info(f'test index is {len(test_index)} in length')
    # df_master = pd.concat([df_master,bin_master], axis =1)

    df_master_train = df_master.loc[train_index]
    df_master_test = df_master.loc[test_index]
    
    logging.info('Get edges')
    overides = training_config.get('band_overides',{})
    tol = training_config.get('bin_tolerance',0.0001)

    edge_map = get_egdes(df_master_train, overides)
    bin_train = build_bins(df_master_train, edge_map, overides)
    feature_profile = get_woe(bin_train)

    logging.info('Apply to test')
    bin_test = build_bins(df_master_test, edge_map, overides)
    print(bin_test.shape)

    bin_train,bin_test, mapping_log, feature_profile_fin, top50_iv = autobinner(bin_train,bin_test, feature_profile, overides, tol)
    assert (list(bin_test) == list(bin_train))
    
    return bin_train, bin_test, edge_map, mapping_log, feature_profile, top50_iv


def autobinner(bin_train, bin_test, feature_profile,overides, tol):
    logging.info('Get profile')
    mapping_log = coarse_class_runner(bin_train, feature_profile, overides, tol)
    bin_train = reclass_df(bin_train, mapping_log )
    feature_profile_fin = get_woe(bin_train)
    top50_iv = list(pd.DataFrame(feature_profile_fin)[['feature', 'iv']].sort_values('iv', ascending=False).head(50)['feature'])
    bin_test = reclass_df(bin_test, mapping_log)
    return bin_train,bin_test, mapping_log, feature_profile_fin, top50_iv


def get_egdes(df, overides):
    edge_map = {}
    for feature in list(df):
        if feature == 'target': continue

        if feature in overides.keys():
            print('**** overide! ****')
            bins,_,_ = extract_overide_info(overides,feature)
            edge_map[feature] = {}
            edge_map[feature]['edges'] = bins
            continue

        if df[feature].nunique() == 2:
            edge_map[feature] = {}
            edge_map[feature]['edges'] = [0, 1]
            continue

        if df[feature].dtype == object:
            edge_map[feature] = {}

            #edge_map[feature]['edges'] = list(df[feature].unique())
            edge_map[feature]= list(df[feature].unique())
            continue

        edge_map[feature] = {}
        try:
            edge_map[feature]['edges'] = get_bin_edges(df[feature])
        except IndexError:
            edge_map.pop(feature)
        except Exception as e:
            logging.ERROR(e)

    return edge_map


def build_bins(df, edge_map, overides={}):
    """
    :param df:
    :param edge_map:
    :return:
    """

    bin_df = pd.DataFrame()
    
    try:
        bin_df['target'] = df.pop('target')
    except:
        pass
    # objects
    
    if overides:
        bin_df = pd.concat([bin_df, apply_overides(df, overides)], axis = 1)

            
    # numeric
    for feature in edge_map.keys():
        if feature in overides.keys():
            continue
        if feature in df.columns[df.dtypes == object]:
            mapper = {}
            for i, edge in enumerate(edge_map[feature]):
                mapper[edge] = f'{i}_{edge}'
            bin_df[feature] = df[feature].map(mapper).fillna('MISSING')
            continue
        try:
            bin_df[feature] = apply_bins(df[feature], edge_map[feature]['edges'])
        except KeyError:
            pass
    return bin_df


def get_numerical_profile(df_master):
    """
    :param df_master:
    :return:
    """
    prof = df_master.describe().T
    prof['dtype'] = df_master.dtypes
    prof['n_zero'] = (df_master == 0).sum()
    prof['n_nan'] = df_master.isna().sum()
    num_profile = prof.T.to_dict(orient='d')
    return num_profile


def get_bin_edges(feature_i: np.array):
    """
    :param feature_i:
    :return:
    """

    col = pd.Series(feature_i)
    # null_idx = col[col.replace({0: np.nan}).isna()].index
    data_idx = col[~col.replace({0: np.nan}).isna()].index
    edges = pd.qcut(col.loc[data_idx], 10, retbins=True, duplicates='drop')[1]
    edges[0] = -np.inf
    edges[-1] = np.inf
    return list(edges)


def apply_bins(feature_i: np.array, edges: list):
    """
    """
    col = pd.Series(feature_i)
    null_idx = col[col.replace({0: np.nan}).isna()].index
    data_idx = col[~col.replace({0: np.nan}).isna()].index
    return pd.concat((
        pd.cut(col.loc[data_idx],
               edges,
               labels=[f'{str(x).zfill(3)}: {np.round(i, 4)}' for x, i in enumerate(edges[1:])]
               )
        , col.loc[null_idx].replace({0: '-1: 0.00', np.nan: '-2: Null'})
    )
        , axis=0).sort_index()


def coarse_classer(df, indexloc_1, indexloc_2):

    if df.Value.dtype != object:
        new_val = pd.DataFrame(np.sum(pd.DataFrame([df.iloc[indexloc_1], df.iloc[indexloc_2]]))).T
    else:
        new_val = pd.DataFrame(np.sum(pd.DataFrame([df.drop('Value',axis=1).iloc[indexloc_1], df.drop('Value',axis=1).iloc[indexloc_2]]))).T

    new_val['Value'] = df['Value'].iloc[indexloc_2]

    mapper = {
        df['Value'].iloc[indexloc_1]: df['Value'].iloc[indexloc_2]
    }

    original = df.drop([indexloc_1, indexloc_2])

    coarsed_df = pd.concat([original, new_val])[['Value', 'All', 'Good', 'Bad']]
    coarsed_df[['All', 'Good', 'Bad']] = coarsed_df[['All', 'Good', 'Bad']].astype(float)
    coarsed_df['Distr_Good'] = coarsed_df['Good'] / coarsed_df['Good'].sum()
    coarsed_df['Distr_Bad'] = coarsed_df['Bad'] / coarsed_df['Bad'].sum()

    coarsed_df['WoE'] = np.log(coarsed_df['Distr_Good'] / coarsed_df['Distr_Bad'])

    coarsed_df = coarsed_df.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    coarsed_df['IV'] = (coarsed_df['Distr_Good'] - coarsed_df['Distr_Bad']) * coarsed_df['WoE']

    coarsed_df = coarsed_df.sort_values(by='Value').reset_index(drop=True)

    return coarsed_df, mapper


def detect_course_class(df, tol = 0.01):
    lst = []
    df['bad_perc'] = df['Bad'] / df['All']
    for i in range(len(df['bad_perc'])):
        for j in range(i + 1, len(df['bad_perc'])):
            if abs(df['bad_perc'].iloc[i] / df['bad_perc'].iloc[j] -1) <= tol and j - i <= 1:
                lst.append(
                    {
                    'index_a': int(i),
                    'index_b': int(j),
                    'difference': abs(df['bad_perc'].iloc[i] / df['bad_perc'].iloc[j] - 1)
                    }
                )

    return pd.DataFrame(lst)


def coarse_class_runner(bin_df, feature_profile, overides, tol = 0.01):
    """

    :param bin_df:
    :param feature_profile:
    :return:
    """
    res_df = pd.DataFrame(feature_profile)
    res_df = res_df.sort_values('iv', ascending=False)
    res_df = res_df.set_index('feature')
    features = res_df.index

    mapping_log = {}
    for feature in features:
        if feature in overides.keys():
            continue
        mapping_log[feature] = {}
        # print(feature, res_df.loc[feature]['iv'])

        iv_df = res_df.loc[feature]['df'].sort_values('Value').copy()
        rep_no = 0
        while 0 == 0:
            s = detect_course_class(iv_df, tol)
            if s.shape == (0, 0):
                if rep_no == 0:
                    mapping_log.pop(feature)
                break
            s = s.sort_values('difference')[['index_a', 'index_b']].iloc[0]
            iv_df, mapper = coarse_classer(iv_df, s['index_a'], s['index_b'])
            mapping_log[feature][rep_no] = mapper
            rep_no += 1
        # print('combining bins makes the IV: {} \n'.format(iv_df['IV'].sum()))

    return mapping_log


def reclass_df(bin_df, mapping_log):
    """

    :param bin_df:
    :param mapping_log:
    :return:
    """

    new_bin_df = bin_df.copy()
    for feat in new_bin_df.columns:
        if feat == 'target': continue
        try:
            replacer = mapping_log[feat]
        except KeyError:
            continue

        for key in replacer.keys():

            new_bin_df[feat] = new_bin_df[feat].replace(replacer[key])

    return new_bin_df


def build_dummies(df, asserted_features=[]):
    """

    :param train:
    :param test:
    :return:
    """

    df_d = pd.DataFrame()
    for feature in list(df):
        bins_df = pd.get_dummies(df[feature], prefix=feature, prefix_sep='___', drop_first=False)
        df_d = pd.concat((df_d, bins_df), axis=1)

    return df_d


def fill_in_test(test_d, train_cols):
    """

    :param test_d:
    :param train_cols:
    :return:
    """
    test_cols = list(test_d)
    cols_to_add = list(set([x for x in train_cols if x not in test_cols]))
    for col in cols_to_add:
        test_d[col] = 0
    return test_d


def train_logistic(train, test, config: dict, list_top_iv):
    """
    :param train:
    :param test:
    :param config:
    :return:
    """
    training_config = config['modelling']
    imp_tol = training_config.get('improvement_tolerance',0.025)
    
    y_train = train['target']
    y_test = test['target']
    current_vars =  []

    # print(current_vars)
    in_model =[]
    feature_list = set(list(train)).intersection(set(list_top_iv))

    train_d = build_dummies(train[feature_list])
    test_d = build_dummies(test[feature_list])
    test_d = fill_in_test(test_d, list(train_d))


    # print(len(feature_list))
    # print(len(list(train)))
    for _ in range(14):

        unilog1 = pd.DataFrame(columns=['feature', 'train_score', 'val_score'])
        in_model_s = []
        for f in current_vars:
            # print(f)
            in_model_s.extend([xyz for xyz in list(train_d) if xyz.startswith(f+'___')])
        if in_model_s:
            clf = LogisticRegression(C=1, penalty='l1', solver='saga')
            clf.fit(train_d[in_model_s], y_train)
            feat = 'current_model'
            score_train = score_model(y_train, clf, train_d[in_model_s], config)
            score_ = score_model(y_test, clf, test_d[in_model_s], config)
            unilog1 = unilog1.append({'feature': feat,
                                      'train_score': score_train,
                                      'val_score': score_
                                     }
                                     , ignore_index=True)
#             print(len(current_vars), roc_auc_score(y_train, clf.predict_proba(train_d[in_model_s])[:, 1]))
            print('\n')
            #         print(unilog1)


        for feat in [x for x in feature_list if x not in current_vars]:
            # logging.info(f'trailling feature {feat}: ')
            in_model = in_model_s.copy()
            to_model = [xyz for xyz in list(train_d) if xyz.startswith(feat+'___')]
            in_model.extend(to_model)
            try:
                clf = LogisticRegression(C=1, penalty='l1', solver='saga', n_jobs = 8)
                clf.fit(train_d[in_model], y_train)
                score_train = score_model(y_train, clf, train_d[in_model], config)
                score_ = score_model(y_test, clf, test_d[in_model], config)
                unilog1 = unilog1.append({'feature': feat,
                                          'train_score': score_train, 
                                          'val_score': score_
                                        }
                                         , ignore_index=True)
                #     print(feat, roc_auc_score(y_test,clf.predict_proba(test_d[in_model])[:,1]))
                logging.info(f'trialling feature {feat} gives score: {score_}')
            except KeyError:
                print(f'Key Error: in {f}')
                print(train[feat].unique())
                feature_list.remove(feat)
                pass
            del in_model

        print(unilog1.sort_values('val_score', ascending=False).head(5))
        try:
            cmod_score = unilog1[unilog1['feature']=='current_model']['val_score'].iloc[0]
        except:
            cmod_score = 0.5
        best_score = unilog1.sort_values('val_score', ascending=False)['val_score'].head(1).values[0]
        if abs(best_score/cmod_score - 1) <= imp_tol:
            print('no improvement automatically')
            break
        else:
            # print(unilog1.sort_values('score', ascending=False)['feature'].head(1).values[0])
            current_vars.append(unilog1.sort_values('val_score', ascending=False)['feature'].head(1).values[0])

    in_model_s = []
    for f in current_vars:
        in_model_s.extend([xyz for xyz in list(train_d) if xyz.startswith(f + '___')])
    clf = LogisticRegression(C=1, penalty='l1', solver='saga')
    clf.fit(train_d[in_model_s], y_train)
    # print(len(in_model_s))
    # print(current_vars, roc_auc_score(y_test, clf.predict_proba(test_d[in_model_s])[:, 1]))
    # print(current_vars, score_model(y_test, clf, test_d[in_model_s], config))

    return clf, in_model_s


def score_model( y_test, model, test, config):

    predictions = model.predict_proba(test)[:, 1]

    predictions_target = pd.DataFrame()
    predictions_target['target'] = y_test
    predictions_target["predictions"] = predictions

    # predictions_target = predictions_target.sort_values(
    #     by=["predictions"], ascending=False
    # )

    score = roc_auc_score(y_test, predictions)
    return score


def train_logistic_fin(train, test, config: dict):
    """
    :param train:
    :param test:
    :param config:
    :return:
    """

    y_train = train.pop('target')
    y_test = test.pop('target')

    in_model = []
    feature_list = []


    for feat in feature_list:
        if train[feat].nunique() == test[feat].nunique():
            pass
        else:
            print(f'removing {feat}')
            feature_list.remove(feat)
            train.pop(feat)
            test.pop(feat)

    train_d = build_dummies(train[feature_list])
    test_d = build_dummies(test[feature_list])
    in_model_s=[]
    for f in feature_list:
        in_model_s.extend([xyz for xyz in list(train_d) if xyz.startswith(f + '___')])
    print(train_d.shape, test_d.shape)

    grid = {'C': [n / 100 for n in range(1, 100)]}

    clf = LogisticRegression(penalty='l1', solver='saga')
    gs = RandomizedSearchCV(clf, grid, 10, cv=3, verbose=3, n_jobs=8, scoring ='roc_auc')
    gs.fit(train_d[in_model_s], y_train)

    clf = LogisticRegression(**gs.best_params_)
    clf.fit(train_d[in_model_s], y_train)

    print(roc_auc_score(y_test, gs.predict_proba(test_d[in_model_s])[:, 1]))

    return clf, in_model_s




def get_woe(bin_df_master, thresh=0.03):
    """
    :param thresh:
    :param bin_df_master:
    :return:
    """

    lst = []
    for col in bin_df_master.columns:
        # print(col)
        if col == 'target':
            continue
        else:
            df, iv = calculate_woe_iv(bin_df_master, col, 'target')
            lst.append({
                'feature': col,
                'iv': iv,
                'df': df
            })

    return lst


def calculate_woe_iv(dataset: pd.DataFrame, feature: str, target: str):
    lst = []
    # print(dataset[feature].nunique())
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })

    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    dset = dset.sort_values(by='Value').reset_index(drop=True)

    return dset, iv


def get_binary(dataset: pd.DataFrame) -> list:
    """
    returns list of binary features (0,1)
    """
    binary_filter = (dataset.max() == 1) & (dataset.nunique() <= 2)
    return dataset.columns[binary_filter].to_list()


def get_predictions(
        test: pd.DataFrame, model, edge_map, mapping_log, model_variables, config
) -> pd.DataFrame:
    """Takes trained model and produces predictions and probability

    Args:
        test (pd.DataFrame):  withheld data set
        model(LGBMClassifier) : Trained model
        config: (dict):  config

    Returns:
        (pd.DataFrame)

    """
    
    training_config = config['modelling']
    targets = training_config['target']
    df_test = test.sort_index().copy()
    overides = training_config.get('band_overides',{})
    bin_test = build_bins(df_test, edge_map,  overides)
    bin_test = reclass_df(bin_test, mapping_log)

    test_d = build_dummies(bin_test)
    test_d = fill_in_test(test_d, model_variables)[model_variables]
    pred_columns = ["predictions", "predictions_proba"]

    df_test["predictions_proba"] = model.predict_proba(test_d[model_variables])[:, 1]
    df_test["predictions"] = model.predict(test_d[model_variables])

    columns_to_grab = (
            [config["id"], config['modelling']['target'][0]]
            + pred_columns
    )

    print(f'VALIDATION SCORE: {roc_auc_score(df_test[targets[0]],df_test["predictions_proba"])}')
    final_df = df_test[columns_to_grab]
    final_df[config["id"]] = final_df[config["id"]].astype(str)
    return final_df

def save_model_coefs(model, model_variables, feature_profile):

    coefs = model.coef_.ravel()
    intercept = model.intercept_

    dfl = pd.DataFrame(feature_profile)
    dc = {k: p for k, p in zip(model_variables, coefs)}
    uni = list(set([x.split('___')[0] for x in dc.keys()]))
    # print(dfl.head())
    out = pd.DataFrame()
    for f in uni:
        # print(f)
        dfn = dfl[dfl['feature'] == f]['df'].iloc[0].copy()
        dfn['feature'] = f
        dfn['feature_value'] = dfn.apply(lambda row: f'{row.feature}___{row.Value}', axis=1)
        out = out.append(dfn)

    out['coef'] = out.feature_value.map(dc).fillna(0)
    out = out.append({'Value': 'intercept', 'coef': intercept[0]}, ignore_index=True)
    return dc, out

def roc_auc_new_set(actual_df, prediction_df):
    actual_df=actual_df [['CUSTID', 'sales']]
    prediction_df = prediction_df[['CUSTID', 'predictions_proba']]

    prediction_df=prediction_df.merge(actual_df, on='CUSTID', how='left')

    prediction_df=prediction_df.fillna(0)
    print(roc_auc_score(prediction_df['sales'], prediction_df['predictions_proba']))

    return prediction_df

def apply_overides(train, overides):
    """
    """
    df_other = train.copy()
    df_overides = pd.DataFrame()
    
    for feature in overides.keys():
        bins, labels, replacer = extract_overide_info(overides, feature)
        col = df_other[feature]
        null_idx  = col[col.replace({0 : np.nan}).isna()].index
        data      = pd.cut(col[~col.index.isin(null_idx)], bins, labels=labels)
        nulls     = col[col.index.isin(null_idx)].replace(replacer)
        df_overides[feature] =  pd.concat([nulls,data], axis = 0).sort_index()
        
    return df_overides

def extract_overide_info(overides, feature):
    """
    """
    
    bins = []
    labels=[]
    replacer={}
    for key, value in overides[feature].items():
        if key not in [None, 0]:
            if key == 'MAX':
                key = np.inf
            bins.append(key)
            labels.append(value)
            
        else:
            if key == None:
                key = np.nan 
            replacer.update({key:value})
    bins.append(-np.inf)
    bins.sort()
    return bins, labels, replacer