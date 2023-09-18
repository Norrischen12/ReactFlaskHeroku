#from modeling_utils import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from retina.metrics import APE
from retina.loading import load_fromS3_convert_clean_save, melt_multiplecols, deal_with_zeros_nans_inDHIS2
import sys
from conf.config import *
sys.path.append(FORECAST_LIB_DIR)
#sys.path.append('../../forecasting-library')
sns.set_style("darkgrid")


def plot_MdAPE(df_pred_actual: pd.DataFrame, thr: int, group_col_name: str,
               pred_col='pred', actual_col='actual', baseline_col='baseline',
               figsize=(14, 6), width_bar=0.6, label_baseline='',
               label_model='', label_improvement='', title='',
               xticks_dict=None):
    """
    plor Median Percentage Absolute Error (MdAPE) seperated by given grouping colmns


    Args:
        df_pred_actual: dataframe including predicted and actual values
        thr: threhold for exluding low values
        group_col_name: groupin column name for which the error is caluculated seperately

    """

    non_zero_res = df_pred_actual[df_pred_actual[actual_col] > thr].reset_index(
        drop=True)
    print('Number of Test Sample {}'.format(len(non_zero_res)))
    non_zero_res['APE'] = APE(non_zero_res[pred_col], non_zero_res[actual_col])
    non_zero_res['APE_base'] = APE(
        non_zero_res[baseline_col], non_zero_res[actual_col])
    base_err = non_zero_res['APE_base'].median()*100
    print(base_err)
    plt.figure(figsize=figsize)
    mape_lst = non_zero_res.groupby(group_col_name)['APE'].median()*100
    x_arr = mape_lst.index*2
    mape_avg = non_zero_res.groupby(group_col_name)['APE_base'].median()*100
    bars = plt.bar(x_arr-width_bar, mape_avg, label=label_baseline,
                   width=width_bar, color='royalblue')

    # access the bar attributes to place the text in the appropriate location
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 1, int(yval))
    bars = plt.bar(x_arr, mape_lst, label=label_model,
                   width=width_bar, color='darkorange')
    # access the bar attributes to place the text in the appropriate location
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 1, int(yval))

    diff = (np.int16(mape_avg) - np.int16(mape_lst))/np.int16(mape_avg)
    bars = plt.bar(x_arr+width_bar, diff*100, label=label_improvement,
                   width=width_bar, color='gray')
    # access the bar attributes to place the text in the appropriate location
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 1, int(yval))

    plt.xticks(x_arr)
    if xticks_dict != None:
        xticks_label = [xticks_dict[i] for i in mape_lst.index]
        plt.gca().set_xticklabels(xticks_label, rotation=0)
    plt.ylim(0, 120)

    plt.legend(bbox_to_anchor=(0., .98, 1., .10), loc='upper left',
               ncol=3, mode="expand", borderaxespad=0.,
               prop={'size': 12})
    plt.ylabel('Median Absolute Percentage Error [%]')
    plt.suptitle(title)


def plot_missing_values(DHIS2_PATH=None, deal_with_false_zeros=False, save=False):
    if DHIS2_PATH == None:
        DHIS2_PATH = 'raw/essential_medicines/johannes_pull_20.08.2020/data.csv'
    to_drop = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1',
               'Unnamed: 0.2']  # , 'Unnamed: 0.1.1.1']
    df_pivot = load_fromS3_convert_clean_save(
        LOCAL_PATH, S3_LINK + DHIS2_PATH, S3_LINK +
        GEOSPATIAL_PATH, product_names, fac_type_list,
        save=False, fname='', to_drop=to_drop)

    if deal_with_false_zeros:
        df_pivot = deal_with_zeros_nans_inDHIS2(df_pivot, product_names,
                                                ops=['invalid_zeros',
                                                     'stock_sum_err',
                                                     'drop_stockout'])

    num_cols = 5
    id_vars = [['date', 'fac_name', 'fac_type', 'latitude',
                'longitude', 'hf_pk']] +\
        [['date', 'fac_name']]*(num_cols-1)
    key_word_valvars = ['Dispensed', '3 Months',
                        '6 Months', 'num_nans', 'num_sample']
    var_name = ['product']*num_cols
    val_name = ['quantity', 'avg_3months_DHIS2',
                'avg_6months_DHIS2', 'num_nans', 'num_sample']

    df = melt_multiplecols(df_pivot, id_vars, key_word_valvars,
                           var_name, val_name, num_cols, product_names)

    plt.figure(figsize=(10, 4))
    date_range = pd.date_range(start=df.date.min(), end=df.date.max())
    q_nans = []
    tot_sample_per_month = len(df_pivot.id.unique())*len(product_names)
    for i in date_range:
        mask = df.date == i
        #print('tot number of samples:', mask.sum())
        if mask.sum() == 0:
            q_nans.append(0)
        else:
            tot_nans = df[mask]['quantity'].isna().sum()
            q_nans.append((
                tot_sample_per_month - mask.sum() + tot_nans)/tot_sample_per_month*100)

    plt.plot(date_range, q_nans)
    plt.ylabel('Percentage of missing values')
    plt.xlabel('Dates')
    if save:
        plt.savefig(LOCAL_PATH+DHIS2_PATH+'.png', dpi=300)
    plt.show()
