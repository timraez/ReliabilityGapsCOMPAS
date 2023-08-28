import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pingouin as pg
from pycm import ConfusionMatrix # see https://www.pycm.io/doc/

import perturbations as pt

# extract dataset information
def prepare_data(dataset_info):
    data = pd.read_csv(dataset_info['filename'])
    target = dataset_info['target']
    attribute = dataset_info['sensitive_attribute']
    Y = data[target]
    Y = Y.replace(to_replace=2, value=0, inplace=False)
    X = data.drop([target], axis=1)
    A = data[attribute]
    return X, A, Y

# generate 5 folds of data
def get_5fold_data(X, A, Y):
    folds = []
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(X)   
    for train_index, test_index in kf.split(X):    
        CV_X = X.iloc[train_index]
        CV_A = A[train_index]
        CV_Y = Y[train_index]
        
        holdout_X = X.iloc[test_index]
        holdout_A = A[test_index]
        holdout_Y = Y[test_index]
        
        fold = {'CV_X': CV_X, 'CV_Y': CV_Y,'holdout_X': holdout_X,'holdout_Y': holdout_Y, 'CV_A': CV_A, 'holdout_A': holdout_A}
        folds.append(fold)
    return folds

# fit five logistic regressions to the five folds
def fit_model(folds, dataset_info):
    for fold in folds:
        CV_X = fold['CV_X']
        CV_Y = fold['CV_Y']
        
        scaler = StandardScaler()
        numerical = dataset_info['numerical_attributes']
        CV_X = CV_X.copy()
        CV_X.loc[:, numerical] = scaler.fit_transform(CV_X[numerical])
        
        lr = LogisticRegression().fit(CV_X, CV_Y)
        fold['model'] = lr
        fold['scaler'] = scaler
    return folds

# fit five models directly from data
def fit_model_data(dataset_info):
    X, A, Y = prepare_data(dataset_info)
    folds = get_5fold_data(X, A, Y)
    folds = fit_model(folds, dataset_info)
    return folds   

# get predictions and probabilites for the five test folds for one noise level
def get_results(folds, proba, dataset_info, parameter_settings):
    var = parameter_settings['variance']
    feature_list_num = parameter_settings['features_num']
    feature_list_cat = parameter_settings['features_cat']
    grouped = parameter_settings['grouped']
    num_minima = parameter_settings['num_minima']
    
    attribute = dataset_info['sensitive_attribute']
    minima_numerical_attributes = dataset_info['minima_numerical_attributes']
    
    outputs_folds = []
    
    for fold in folds:
        outputs = {}
        
        holdout_X = fold['holdout_X']
        holdout_A = fold['holdout_A']
        
        holdout_X = holdout_X.copy()
        holdout_X_perturbed = holdout_X.copy()
        
        pt.perturb_total(holdout_X_perturbed, attribute, feature_list_num, feature_list_cat, var, proba, grouped, num_minima, minima_numerical_attributes)
        
        scaler = fold['scaler']        
        lr = fold['model']
        numerical = dataset_info['numerical_attributes']
        
        holdout_X.loc[:, numerical] = scaler.transform(holdout_X[numerical])
        holdout_X_perturbed.loc[:, numerical] = scaler.transform(holdout_X_perturbed[numerical])
        
        probabilities = lr.predict_proba(holdout_X)[:, 1]     
        predictions = lr.predict(holdout_X)

        probabilities_perturbed = lr.predict_proba(holdout_X_perturbed)[:, 1]
        predictions_perturbed = lr.predict(holdout_X_perturbed)
        
        outputs['preds'] = predictions
        outputs['probs'] = probabilities
        outputs['preds_p'] = predictions_perturbed
        outputs['probs_p'] = probabilities_perturbed       
       
        outputs[attribute] = holdout_A.to_numpy()
        
        outputs = pd.DataFrame(outputs)
        outputs_folds.append(outputs)
    
    return outputs_folds

# calculate irr metrics on five folds for one noise level, then compute mean of five folds
def get_metrics(outputs_folds, dataset_info):
    
    attribute = dataset_info['sensitive_attribute']
    
    metrics_folds = []
    
    for outputs in outputs_folds:
        
        metrics = {}
        
        # get group-filtered outputs
        out_g0 = outputs[outputs[attribute]==0]
        out_g1 = outputs[outputs[attribute]==1]
        
        # transform data into long form for ICC computation with pingouin package
        probas_group_0 = out_g0[['probs', 'probs_p']].copy()
        probas_group_1 = out_g1[['probs', 'probs_p']].copy()
        probas_group_0['index'] = probas_group_0.index
        probas_group_1['index'] = probas_group_1.index      
        probas_group_0 = pd.melt(probas_group_0, id_vars=['index'], value_vars=list(probas_group_0)[:-1])
        probas_group_1 = pd.melt(probas_group_1, id_vars=['index'], value_vars=list(probas_group_1)[:-1])
        
        # compute ICC statistics using pingouin (yields table)
        icc_group_0 = pg.intraclass_corr(data=probas_group_0, targets='index', raters='variable', ratings='value')
        icc_group_1 = pg.intraclass_corr(data=probas_group_1, targets='index', raters='variable', ratings='value')
        
        # ICC 2
        metrics['group_0_ICC_2'] = icc_group_0['ICC'][1]
        metrics['group_1_ICC_2'] = icc_group_1['ICC'][1]
        
        # get confusion matrix for the kappa metrics
        cm_0 = ConfusionMatrix(out_g0['preds'].to_numpy(), out_g0['preds_p'].to_numpy())
        cm_1 = ConfusionMatrix(out_g1['preds'].to_numpy(), out_g1['preds_p'].to_numpy()) # have deleted option digit=5
        
        # Cohen's Kappa
        metrics['group_0_kappa'] = cm_0.Kappa
        metrics['group_1_kappa'] = cm_1.Kappa
        
        # intermediate results for prevalence and bias adjusted kappa, based on Byrt et al.
        group_0_PI = (cm_0.TP[1]-cm_0.TN[1])/cm_0.POP[1]
        group_1_PI = (cm_1.TP[1]-cm_1.TN[1])/cm_1.POP[1]
        group_0_BI = (cm_0.FP[1]-cm_0.FN[1])/cm_0.POP[1]
        group_1_BI = (cm_1.FP[1]-cm_1.FN[1])/cm_1.POP[1]

        # PABAK (prevalence adjusted bias adjusted kappa), based on Byrt et al.
        metrics['group_0_PABAK'] = cm_0.Kappa*(1-group_0_PI**2+group_0_BI**2) + group_0_PI**2 - group_0_BI**2
        metrics['group_1_PABAK'] = cm_1.Kappa*(1-group_1_PI**2+group_1_BI**2) + group_1_PI**2 - group_1_BI**2

        # PAK (prevalence adjusted kappa), based on Byrt. et al.
        metrics['group_0_PAK'] = cm_0.Kappa*(1-group_0_PI**2) + group_0_PI**2
        metrics['group_1_PAK'] = cm_1.Kappa*(1-group_1_PI**2) + group_1_PI**2  

        # BAK (bias adjusted kappa), based on Byrt et al.
        metrics['group_0_BAK'] = cm_0.Kappa*(1+group_0_BI**2) - group_0_BI**2
        metrics['group_1_BAK'] = cm_1.Kappa*(1+group_1_BI**2) - group_1_BI**2
        
        metrics_folds.append(metrics)
    
    mean_metrics = pd.DataFrame(metrics_folds).mean(axis=0)
        
    return mean_metrics

# compute probabilities and predictions for array of noise levels on five folds
def get_results_probas(folds, dataset_info, parameter_settings):
    
    print('Computing outputs...')
    
    probas = parameter_settings['probabilities']
    
    outputs_probas = {}
    
    for i in probas:
        i = round(i, 5)
        outputs_folds = get_results(folds, i, dataset_info, parameter_settings)
        outputs_probas[i] = outputs_folds
        
    print('Done.')
    
    return outputs_probas

# compute irr metrics for array of noise levels
def get_metrics_probas(outputs_probas, dataset_info):
    
    metrics_probas = {}
    
    for i in outputs_probas:
        print(f'Computing metrics with proba {i} ...')
        mean_metrics = get_metrics(outputs_probas[i], dataset_info)
        metrics_probas[i] = mean_metrics 
    
    metrics_probas = pd.DataFrame(metrics_probas).T

    print('Done.')

    return metrics_probas

# get irr metrics directly from data
def get_metrics_probas_from_model(dataset_info, parameter_settings, folds):
    outputs_probas = get_results_probas(folds, dataset_info, parameter_settings)
    metrics_probas = get_metrics_probas(outputs_probas, dataset_info)  
    return metrics_probas
 