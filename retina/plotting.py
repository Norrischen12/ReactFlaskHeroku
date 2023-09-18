import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from retina.modeling import *
from retina.metrics import *
import sys
sys.path.append('../')
sns.set_style("darkgrid")


def plot_MdAPE(df_pred_actual: pd.DataFrame, thr: int, group_col_name: str,
               pred_col='pred', actual_col='actual', baseline_col='baseline',
               figsize=(14, 6), width_bar=0.6, label_baseline='',
               label_model='', label_improvement='', title='',
               xticks_label=None, rotation=0):
    """
    plot Median Percentage Absolute Error (MdAPE) seperated by given grouping columns


    Args:
        df_pred_actual: dataframe including predicted and actual values
        thr: threshold for excluding low values
        group_col_name: grouping column name for which the error is calculated separately

    """
    plt.figure(figsize=figsize)
    non_zero_res = df_pred_actual[df_pred_actual[actual_col] > thr].reset_index(drop=True)
    print('Number of Test Sample {}'.format(len(non_zero_res)))

    non_zero_res['APE'] = APE(non_zero_res[pred_col], non_zero_res[actual_col])
    non_zero_res['APE_base'] = APE(non_zero_res[baseline_col], non_zero_res[actual_col])
    base_err = non_zero_res['APE_base'].median()*100


    plt.figure(figsize=figsize)
    mape_lst = non_zero_res.groupby(group_col_name)['APE'].median()*100
    x_arr_map = dict(zip(range(len(mape_lst)), mape_lst.index))
    x_arr = np.array(list(x_arr_map.keys()))*2#mape_lst.index*2
    mape_avg = non_zero_res.groupby(group_col_name)['APE_base'].median()*100
    bars = plt.bar(x_arr-width_bar, mape_avg, label=label_baseline, width=width_bar, color='royalblue')

    # access the bar attributes to place the text in the appropriate location
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 1, int(yval))
    bars = plt.bar(x_arr, mape_lst, label=label_model, width=width_bar, color='darkorange')
    # access the bar attributes to place the text in the appropriate location
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 1, int(yval))

    diff = (np.int16(mape_avg) - np.int16(mape_lst))/np.int16(mape_avg)
    bars = plt.bar(x_arr+width_bar, diff*100, label=label_improvement, width=width_bar, color='gray')
    # access the bar attributes to place the text in the appropriate location
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 1, int(yval))

    plt.xticks(x_arr)
    if xticks_label == None:
        xticks_label = list(x_arr_map.values())
    plt.gca().set_xticklabels(xticks_label, rotation=rotation)
    plt.ylim(0, 120)

    plt.legend(bbox_to_anchor=(0., .98, 1., .10), loc='upper left', ncol=3, mode="expand", borderaxespad=0.,
               prop={'size': 12})
    plt.ylabel('Median Absolute Percentage Error [%]')
    plt.suptitle(title)



