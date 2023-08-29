import matplotlib.pyplot as plt

# five dictionaries with figure type information (kappa, pabak, icc2, bak and pak corrections for two groups)

figure_kappa = {
    'tag': 'Kappa', 
    'group_0_metric': 'group_0_kappa', 
    'group_1_metric': 'group_1_kappa',
    'y_label': "Cohen's Kappa"
}

figure_PABAK = {
    'tag': 'PABAK', 
    'group_0_metric': 'group_0_PABAK', 
    'group_1_metric': 'group_1_PABAK',
    'y_label': "Byrt's PABAK"
}

figure_ICC2 = {
    'tag': 'ICCA', 
    'group_0_metric': 'group_0_ICC_2', 
    'group_1_metric': 'group_1_ICC_2',
    'y_label': "ICC(A,1)"
}  

figure_correction_group_0 = {
    'tag': 'CORR_G0',
    'group_CK': 'group_0_kappa',
    'group_PABAK': 'group_0_PABAK',
    'group_PAK': 'group_0_PAK',
    'group_BAK': 'group_0_BAK',
    'y_label': 'BI and PI Correction Group 0'
}

figure_correction_group_1 = {
    'tag': 'CORR_G1',
    'group_CK': 'group_1_kappa',
    'group_PABAK': 'group_1_PABAK',
    'group_PAK': 'group_1_PAK',
    'group_BAK': 'group_1_BAK',
    'y_label': 'BI and PI Correction Group 1'
}

# plot and save figures with two metrics (kappa, pabak, icc)
def plot_save_figure_2_metrics(metrics_probas, parameter_settings, figure_type):
    tag = figure_type['tag']
    fig = plt.figure()
    subfig = fig.add_subplot(1, 1, 1)
    subfig.plot(metrics_probas[figure_type['group_0_metric']], color = 'blue', alpha = 0.5,   label = 'Group 0')
    subfig.plot(metrics_probas[figure_type['group_1_metric']], color = 'orange', label = 'Group 1')
    subfig.set_ylabel(figure_type['y_label'])
    fig_label = f'feature set={parameter_settings["feature_set"]}, var={parameter_settings["variance"]}, grouped={parameter_settings["grouped"]}, num_min={parameter_settings["num_minima"]}'
    subfig.set_xlabel(f'noise level ({fig_label})')
    subfig.legend()
    fig.savefig(f'figures/{tag}_feature_set={parameter_settings["feature_set"]}_var={parameter_settings["variance"]}_grouped={parameter_settings["grouped"]}_num_min={parameter_settings["num_minima"]}.png', bbox_inches = "tight") 
    
#plot and save figure with four metrics (comparison of kappa and pabak, separately for groups)
def plot_save_figure_corr(metrics_probas, parameter_settings, figure_type):
    tag = figure_type['tag']
    fig = plt.figure()
    subfig = fig.add_subplot(1, 1, 1)
    subfig.plot(metrics_probas[figure_type['group_CK']], color='orange', label = 'CK')
    subfig.plot(metrics_probas[figure_type['group_PABAK']], color = 'blue', alpha = 0.5, label='PABAK')
    subfig.plot(metrics_probas[figure_type['group_PAK']], color = 'k', linestyle = 'dashed', label='PI-corrected')
    subfig.plot(metrics_probas[figure_type['group_BAK']], color = 'k', linestyle = 'dotted',  label='BI-corrected')
    subfig.set_ylabel(figure_type['y_label'])
    fig_label = f'feature set={parameter_settings["feature_set"]}, var={parameter_settings["variance"]}, grouped={parameter_settings["grouped"]}, num_min={parameter_settings["num_minima"]}'
    subfig.set_xlabel(f'noise level ({fig_label})')
    subfig.legend()
    fig.savefig(f'figures/{tag}_feature_set={parameter_settings["feature_set"]}_var={parameter_settings["variance"]}_grouped={parameter_settings["grouped"]}_num_min={parameter_settings["num_minima"]}.png', bbox_inches = "tight") 

#plot all figures at once
def plot_all_figures(metrics_probas, parameter_settings):
    for figure_type_2 in [figure_kappa, figure_PABAK, figure_ICC2]:
        plot_save_figure_2_metrics(metrics_probas, parameter_settings, figure_type_2)
    for figure_type_corr in [figure_correction_group_0, figure_correction_group_1]:
        plot_save_figure_corr(metrics_probas, parameter_settings, figure_type_corr)
