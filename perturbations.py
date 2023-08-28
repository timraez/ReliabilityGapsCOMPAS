import numpy as np

# generate perturbation vector for categorial features
def get_perturbation_cat(data, proba):
    return np.random.choice([0, 1], size=len(data), p=[(1-proba), proba])

# perturb single categorial feature
def perturb_cat(data, feature, proba):
    data[feature] = np.add(data[feature],get_perturbation_cat(data, proba)) % 2
    return data

# perturb single categorial feature for one group
def perturb_cat_group(data, attribute, attribute_value, feature, proba):
    data.loc[data[attribute]==attribute_value, [feature]] = perturb_cat(data.loc[data[attribute]==attribute_value, [feature]], feature, proba)
    return data

# perturb set of categorical features, controlling for noise distribution over groups
def perturb_cat_grouped(data, attribute, feature_list, proba):
    for feature in feature_list:
        perturb_cat_group(data, attribute, 1, feature, proba)
        perturb_cat_group(data, attribute, 0, feature, proba)
    return data

# perturb set of categorical features, no control of noise distribution over groups
def perturb_cat_ungrouped(data, feature_list, proba):
    for feature in feature_list:
        perturb_cat(data, feature, proba)
    return data

# generate perturbation vector for numerical features
def get_perturbation_num(data, var, proba):
    return np.random.choice([0, 1], size=len(data), p=[(1-proba), proba])*np.random.normal(0, var, size=(len(data)))

# perturb single numerical feature, either with or withtout minimal value
def perturb_num(data, feature, var, proba, num_minima, minima_numerical_attributes):
    if num_minima == 'Y':
        minim = minima_numerical_attributes[feature]
        data[feature] = np.clip(np.rint(np.add(data[feature], get_perturbation_num(data, var, proba))), a_min = minim, a_max = None)    
    elif num_minima == 'N':
        data[feature] = np.rint(np.add(data[feature], get_perturbation_num(data, var, proba)))
    else: print('Warning: Choose either "Y" or "N" for parameter "num_minima"...')
    return data

# perturb single numerical feature for one group
def perturb_num_group(data, attribute, attribute_value, feature, var, proba, num_minima, minima_numerical_attributes):
    data.loc[data[attribute]==attribute_value, [feature]] = perturb_num(data.loc[data[attribute]==attribute_value, [feature]], feature, var, proba, num_minima, minima_numerical_attributes)
    return data

# perturb set of numerical features, controlling for noise distribution over groups
def perturb_num_grouped(data, attribute, feature_list, var, proba, num_minima, minima_numerical_attributes):
    for feature in feature_list:
        perturb_num_group(data, attribute, 1, feature, var, proba, num_minima, minima_numerical_attributes)         
        perturb_num_group(data, attribute, 0, feature, var, proba, num_minima, minima_numerical_attributes)
    return data

# perturb set of numerical features, no control of noise distribution
def perturb_num_ungrouped(data, feature_list, var, proba, num_minima, minima_numerical_attributes):
    for feature in feature_list:
        perturb_num(data, feature, var, proba, num_minima, minima_numerical_attributes)
    return data

# perturb both numerical and categorial features, grouped or ungrouped
def perturb_total(data, attribute, feature_list_num, feature_list_cat, var, proba, grouped, num_minima, minima_numerical_attributes):
    if grouped == 'Y':
        perturb_num_grouped(data, attribute, feature_list_num, var, proba, num_minima, minima_numerical_attributes)
        perturb_cat_grouped(data, attribute, feature_list_cat, proba)
    elif grouped == 'N': 
        perturb_num_ungrouped(data, feature_list_num, var, proba, num_minima, minima_numerical_attributes)
        perturb_cat_ungrouped(data, feature_list_cat, proba)
    else: print('Warning: Choose either "Y" or "N" for parameter "grouped"...')