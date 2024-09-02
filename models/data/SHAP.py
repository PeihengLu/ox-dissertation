# performing SHAP analysis on the model
from os.path import join as pjoin
import os
import sys
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
# random decision forest regressor
from xgboost import XGBRegressor
# use agg backend to prevent the figure from showing
plt.switch_backend('agg')

sys.path.append(pjoin(os.path.dirname(__file__), '../../'))

from utils.data_utils import convert_to_SHAP, convert_to_shap_1bp
from utils.stats_utils import get_pearson_and_spearman_correlation

def SHAP_analysis(input_data, output, reset=False, fig_size=(10, 6), max_display=10):
    data_path = output
    shap_values_path = f"{output.split('.')[0]}.npy"

    # if the data is already parsed, load it
    if os.path.exists(shap_values_path) and not reset:
        # load up the shap values if they exist
        shap_values = np.load(shap_values_path, allow_pickle=True)
        data = pd.read_csv(data_path)        

        # split the data into features and targets
        target = data['editing-efficiency']
        features = data.drop(columns=['editing-efficiency', 'group-id'])
    else:
        if not os.path.exists(data_path) or reset: 
            convert_to_shap_1bp(input_data)
        data = pd.read_csv(data_path) 
        group_ids = data['group-id']       

        # split the data into features and targets
        target = data['editing-efficiency']
        features = data.drop(columns=['editing-efficiency', 'group-id'])

        train_size = int(0.8 * len(data))
        train_features = features.iloc[:train_size,:]
        train_target = target.iloc[:train_size]
        test_features = features.iloc[train_size:]
        test_target = target.iloc[train_size:]

        print('Training the model...')

        # train the model
        model = XGBRegressor()
        model.fit(train_features, train_target)

        # test the model and calculate the Pearson's R and Spearman's correlation
        test_output = model.predict(test_features)
        pearson, spearman = get_pearson_and_spearman_correlation(test_target, test_output)

        print(f'Pearson\'s R: {pearson}')
        print(f'Spearman\'s correlation: {spearman}')

        print('Explaining the model...')
        # explain the model
        explainer = shap.Explainer(model)
        shap_values = explainer(features)
        shap_values = shap_values.values

        # reorder the features in the csv file according to the mean absolute shap values
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        features = features.iloc[:, sorted_indices]
        # join features with the target
        data = pd.concat([features, group_ids, target], axis=1)
        data.to_csv(data_path, index=False)

        # reorder the shap values as well
        shap_values = shap_values[:, sorted_indices]
        
        # save the shap values
        np.save(shap_values_path, shap_values)

    print('Plotting the SHAP values...')

    # plot the SHAP values  
    fig = plot_shap_values(shap_values, features, max_display=max_display, figsize=fig_size)
    figure_name = f'{os.path.basename(output).split(".")[0]}.png'
    plt.savefig(f'../../dissertation/figures/{figure_name}', dpi=300)
    # clear all figures
    plt.close('all')

def plot_shap_values(shap_values: np.ndarray, features, max_display=10, figsize=(14, 6)) -> plt.Figure:
    '''
    shap_values is of size (n_samples, n_features)
    '''
    plt.tight_layout()


    # create a summary plot
    shap.summary_plot(shap_values, features, max_display=max_display, show=False, plot_size=figsize)
    fig, ax = plt.gcf(), plt.gca()
    
    # change the color palette
    for fc in fig.get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                fcc.set_cmap(sns.color_palette("vlag", as_cmap=True))
    
    # adjust the plot size
    fig.set_size_inches([figsize[0]-2.5, figsize[1]+5])

    # replace melting-temperature with mt, minimum-free-energy with mfe, gc-content with gcc in the y labels
    y_labels = ax.get_yticklabels()
    for label in y_labels:
        label.set_text(label.get_text().replace('melting-temperature', 'tm').replace('minimum-free-energy', 'mfe').replace('gc-content', 'gcc').replace('maximal-length-of-consecutive-a-sequence', 'max-cas').replace('maximal-length-of-consecutive-t-sequence', 'max-cts').replace('maximal-length-of-consecutive-c-sequence', 'max-ccs').replace('maximal-length-of-consecutive-g-sequence', 'max-cgs').replace('edit-type', 'et').replace('-at-protospacer-position-', '-ap-').replace('gc-count', 'gc#').replace('before-edit-position', 'bep').replace('after-edit-position', 'aep'))
        # remove hyphens
        label.set_text(label.get_text().replace('-', ' '))

    # set the y labels
    ax.set_yticklabels(y_labels)
    
    if figsize[0] <= 10:
        font_size = 26
        print('Rotating y labels')
        plt.yticks(rotation=20)
        # remove the color bar
        fig.axes[1].remove()
        # showing x ticks in step of 30
        x_ticks = ax.get_xticks()
        x_ticks_min = x_ticks[0]
        x_ticks_max = x_ticks[-1]
        x_ticks = np.arange(x_ticks_min, x_ticks_max, 30)
        ax.set_xticks(x_ticks)
    else:
        font_size = 30
        # Get colorbar
        cb_ax = fig.axes[1] 

        # Modifying color bar parameters
        cb_ax.tick_params(labelsize=font_size)
        cb_ax.set_ylabel("Feature Value", fontsize=font_size)
        
    # increase font size for all text
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
        
    plt.xlabel('SHAP Value', fontsize=font_size)
    # rotate y labels if height > width
    print('Figure size:', figsize)

    return fig

if __name__ == '__main__':
    reset = False
    # SHAP_analysis('std/std-dp-hek293t-pe2.csv', 'shap/shap-dp-hek293t-pe2.csv', reset=reset, fig_size=(20, 8), max_display=24)
    # perform SHAP analysis on different data types
    # SHAP_analysis('std/std-pd-hek293t-pe2-replace.csv', 'shap/shap-pd-hek293t-pe2-replace.csv', reset=reset, fig_size=(8, 3.5), max_display=10)
    # SHAP_analysis('std/std-pd-hek293t-pe2-insert.csv', 'shap/shap-pd-hek293t-pe2-insert.csv', reset=reset, fig_size=(8, 3.5), max_display=10)
    # SHAP_analysis('std/std-pd-hek293t-pe2-delete.csv', 'shap/shap-pd-hek293t-pe2-delete.csv', reset=reset, fig_size=(8, 3.5), max_display=10)
    # for data_source in ['adv-pe2', 'k562-pe2', 'k562mlh1dn-pe2', 'hek293t-pe2']:    
    #     SHAP_analysis(f'std/std-pd-{data_source}.csv', f'shap/shap-pd-{data_source}.csv', reset=reset, fig_size=(9, 6), max_display=24)
    for edit in ['replace', 'insert', 'delete']:
        for data_source in ['dp_small-a549-pe2max']:    
            SHAP_analysis(f'std/std-{data_source}.csv', f'shap/shap_1bp-{data_source}-{edit}.csv', reset=reset, fig_size=(9, 6), max_display=24)