import scipy.stats as st
import scs
from flask import jsonify
import scipy
import time
import cvxpy as cp
import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import os
import argparse
import logging
import numpy as np
from datetime import *
from retina.feature_engineering import (
    add_rolling,
    add_prev_periods,
    split_dates,
    str_to_cat,
    add_deriv,
    create_label)
import pandas as pd
from retina.metrics import *
from src.create_features import create_features_essential_meds
from src.preprocess import fetch_preprocessedDHIS2_EssMeds
from src.predict_utilization import predict_utilization
from conf.config import *
from src.model import get_training_data, train_model
import sys
sys.path.append(FORECAST_LIB_DIR)


def CleanDhis2(data, product_names):
    # split 'name' column into multiple columns
    fname = f'tmp/dhis2Clean.csv'
    if not os.path.isfile(fname):
        df_raw = data
        for i in range(0, len(product_names)):
            if i == 0:
                df = df_raw[df_raw['Data_name'].str.contains(
                    product_names[i], regex=False)]
                df['product'] = product_names[i]
                print(product_names[i], len(df))
            else:
                tmp = df_raw[df_raw['Data_name'].str.contains(
                    product_names[i], regex=False)]
                tmp['product'] = product_names[i]
                print(product_names[i], len(tmp))
                df = pd.concat([df, tmp])
        data = df
        data['date'] = data['year'].astype(
            str) + '-' + data['month'].astype(str) + '-01'
        data['date'] = pd.to_datetime(data['date'])
        data['status'] = np.where(data['Data_name'].str.contains("Opening"), "Opening Balance (A)",
                                  np.where(data['Data_name'].str.contains("Received"), "Quantity Received (B)",
                                           np.where(data['Data_name'].str.contains("Dispensed"), "Quantity Dispensed (D)",
                                                    np.where(data['Data_name'].str.contains("Losses"), "Losses / Adjustments (C)",
                                                             np.where(data['Data_name'].str.contains("Closing"), "Closing Balance (E)",
                                                                      np.where(data['Data_name'].str.contains("Days"), "Days Out of Stock (F)",
                                                                               np.where(data['Data_name'].str.contains("Stockout"), "Stockout", "AMC")))))))
        AllEMshape = df.pivot_table(
            index=['Organisation unit', 'date', 'product'], columns='status', values="Value").reset_index()

        # Creating new columns based on conditions
        AllEMshape['zero'] = np.where((AllEMshape[('Opening Balance (A)')] == 0) &
                                      (AllEMshape[('Quantity Received (B)')] == 0) &
                                      (AllEMshape[('Closing Balance (E)')] == 0) &
                                      (AllEMshape[('Quantity Dispensed (D)')] == 0) &
                                      (AllEMshape[('Losses / Adjustments (C)')] == 0), 1, 0)

        AllEMshape['balance'] = np.where((AllEMshape[('Opening Balance (A)')] +
                                          AllEMshape[('Quantity Received (B)')] -
                                          AllEMshape[('Quantity Dispensed (D)')] +
                                          AllEMshape[('Losses / Adjustments (C)')] ==
                                          AllEMshape[('Closing Balance (E)')]), 0, 1)

        AllEMshape['stockout'] = np.where((AllEMshape[('Days Out of Stock (F)')] > 0) |
                                          (AllEMshape[('Stockout')] == 1), 1, 0)

        Allmiss = AllEMshape[(AllEMshape['zero'] == 0) & (
            AllEMshape['balance'] == 0) & (AllEMshape['stockout'] == 0)]

    else:
        Allmiss = pd.read_csv(fname)

    return Allmiss


def ProcessBFeature(data):
    facFeature = pd.read_csv(
        "/Users/angelchung/Desktop/2022Q4/MFLhfpk_Dhis2AllQ4.csv")
    budget = pd.read_csv(
        "/Users/angelchung/Desktop/2023Q2/tmp/2023Q2Budget.csv")
    budget = budget[['name2', 'product', 'stock', 'LevelCare']]
    budget['product'] = budget['name2']
    data['SOH'] = data['Closing Balance (E)']
    data['quantity'] = data['Quantity Dispensed (D)']
    data.rename(
        columns={'Organisation unit': 'organisationunit_id'}, inplace=True)
    data = data.merge(facFeature, how='left', on='organisationunit_id')
    data = data.merge(budget, how='left', on='product')
    Dhis2BFeature = data[['product', 'organisationunit_id', 'hf_pk', 'date',
                          'quantity', 'facility_type', 'lat', 'long', 'district', 'SOH']]
    Dhis2BFeature = Dhis2BFeature.dropna()
    return Dhis2BFeature


def create_features_essential_meds(df: pd.DataFrame,
                                   grouping_cols=['hf_pk', 'product'],
                                   date_column='date',
                                   lead_time=1,
                                   return_mapping=False) -> pd.DataFrame:
    """create features using forecasting library
    Args:
        df: PREPROCESSED dataframe
    Returns: df with ts features
    """

    df = df.copy()
    quantity_column = 'quantity'
    df = add_rolling(df, date_column, grouping_cols,
                     quantity_column, [2, 3, 4, 5, 6], rolling_stat='mean')
    df = add_prev_periods(df, date_column, grouping_cols,
                          quantity_column, 6)
    df = add_rolling(df, date_column, grouping_cols,
                     quantity_column, [3, 6], rolling_stat='std')
    df = add_rolling(df, date_column, grouping_cols,
                     quantity_column, output_name='total_sample',
                     rolling_stat='count')
    df = add_rolling(df, date_column, grouping_cols,
                     quantity_column,  [3, 6, 12],
                     rolling_stat='count')
    print('cols features: ', df.columns)
    df = split_dates(df, date_column)

    df = add_deriv(df, date_column, grouping_cols,
                   quantity_column, 3)
    df = add_rolling(df, date_column, ['product'],
                     quantity_column, [2, 3, 4, 5, 6, 10], output_name='avg_per_product')
    # df = df.fillna(0)  # fill standard deviation with 0 when missing

    if return_mapping:
        df, mapping = str_to_cat(df, return_mapping=True)
    else:
        df = str_to_cat(df)
    df = create_label(df, date_column, grouping_cols,
                      target_column=quantity_column, lead_time=lead_time, mode='test')
    # adding to our date the total lead time!
    df['date'] = df.date + pd.DateOffset(months=lead_time)

    if return_mapping:
        return df, mapping
    return df


def get_predictions(dates, n_estimators, date_column, target_col, lead_time, df_weight=None, aggregation=None):
    print('constructing dataset')
    fname = f'tmp/df4ML8_2023.csv'
    if not os.path.isfile(fname):  # change to our own dhis2 pipeline
        df = fetch_preprocessedDHIS2_EssMeds(
            preprocessed_file_name='pivot_from_raw_DHIS2')
        if aggregation == None:
            # create features
            df4ML, mapping = create_features_essential_meds(
                df, lead_time=lead_time, return_mapping=True)

        elif aggregation == 'district':
            feat_agg = ['date', 'district', 'product', 'avg_3months_DHIS2',
                        'quantity', 'num_fac_per_district']
            df = df.merge(df.groupby('district')[
                'fac_name'].nunique().reset_index().rename(columns={
                    'fac_name': 'num_fac_per_district'}), on='district')[feat_agg]
            df = df.groupby(['date', 'district',
                             'product', 'num_fac_per_district'])[[
                                 'quantity', 'avg_3months_DHIS2']].mean().astype('int').reset_index()
            output_columns = [
                i for i in output_columns if i != 'hf_pk'] + ['district']
            # create features
            df4ML, mapping = create_features_essential_meds(df, grouping_cols=[aggregation, 'product'],
                                                            lead_time=lead_time, return_mapping=True)
        else:
            raise ValueError(f'{aggregation} method is not implemented!')
        df4ML.to_csv(fname, index=False)

    df4ML = pd.read_csv(fname)
    drop_cols = ['SOH']
    # drop_cols = ['avg_3months_DHIS2', 'avg_6months_DHIS2','log_dif']
    df4ML = df4ML.drop(drop_cols, axis=1)
    df = df4ML.copy()
    df['date'] = df['date'].map(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    for i in range(len(dates)):
        Tdate = dates[i]
        print('date', Tdate)
        # df['date']=df['date'].map(lambda x:datetime.strptime(x,"%Y-%m-%d"))
        train = df[(df['date'] + pd.offsets.DateOffset(months=(lead_time-1))) < Tdate]
        train = train[train['district'] != 0].copy()
        if train.isna().sum().sum() > 0:
            train = train.dropna()
            warnings.warn('data include Nan values. Dropped from train sets!')
        print(f"Training min: {train['date'].min()}")
        print(f"Training max: {train['date'].max()}")
        val = df[df['date'] == Tdate]
        val = val.dropna()
        print(f"Test min: {val['date'].min()}")
        print(f"Test max: {val['date'].max()}")
        print(f"Total sample test: {len(val)}")
    if not df_weight is None:
        df_weight['date'] = df_weight['date'].astype('datetime64[ns]')
        train = pd.merge(train, df_weight, on=[
                         'hf_pk', 'date', 'product'], how='left')
        train['weight'] = train['weight'].fillna(1)
    rfr = train_model(train, n_estimators)
    xs_train, _, _ = get_training_data(train)
    xs_test, _, _ = get_training_data(val)
    for i, tree in enumerate(rfr.estimators_):
        train[f'demand{i}'] = np.maximum(tree.predict(xs_train), 0)
        val[f'demand{i}'] = np.maximum(tree.predict(xs_test), 0)
    return train, val


# save allocations for given demand prediction approach
def get_allocation(df, n_estimators, allocation_max, date, product, optimize_fn):
    df = df[(df['product'] == product) & (df['date'] == date)].copy()
    allocation_max = df['budget'].unique()
    allocation = optimize_fn(df, n_estimators, allocation_max, product)
    df['allocation'] = allocation
    return df


# optimize allocations using linear programming
def optimize_lp(demand, allocation_max, product):
    # variables
    n_facilities, n_samples = demand.shape
    print(demand.shape)

    if np.shape(demand)[0] != 0:
        opt_type = 'none'
        demand_multiplier = 1.0

        # linear program
        allocation = cp.Variable(shape=(n_facilities), name="allocation")
        loss = cp.Variable(shape=(n_facilities, n_samples), name="loss")
        constraints = [allocation >= 0, cp.sum(
            allocation) <= allocation_max, loss >= 0]
        for i in range(n_samples):
            constraints += [loss[:, i] >= demand[:, i]
                            * demand_multiplier - allocation]

        if opt_type == 'cvar_samples':
            loss_cvar = cp.Variable(shape=(n_samples), name="loss_cvar")
            z_cvar = cp.Variable(name="z_cvar")
            for i in range(n_samples):
                constraints += [loss_cvar[i] >= cp.sum(loss[:, i]) - z_cvar]
                constraints += [loss_cvar[i] >= 0]
            objective = cp.Minimize(
                z_cvar + 1.1 * cp.sum(loss_cvar) / n_samples)
        elif opt_type == 'cvar_facilities':
            loss_cvar = cp.Variable(shape=(n_facilities), name="loss_cvar")
            z_cvar = cp.Variable(name="z_cvar")
            for i in range(n_facilities):
                constraints += [loss_cvar[i] >= cp.sum(loss[i, :]) - z_cvar]
                constraints += [loss_cvar[i] >= 0]
            objective = cp.Minimize(
                z_cvar + 1.1 * cp.sum(loss_cvar) / n_facilities)
        elif opt_type == 'quad':
            objective = cp.Minimize(cp.sum_squares(loss))
        else:
            objective = cp.Minimize(cp.sum(loss))

        # solution
        prob = cp.Problem(objective, constraints)
        if opt_type == 'quad':
            try:
                prob.solve()
            except:
                print('error with quadratic, trying linear')
                objective = cp.Minimize(cp.sum(loss))
                prob = cp.Problem(objective, constraints)
                prob.solve(cp.SCIPY, scipy_options={"method": "highs"})
        else:
            prob.solve(cp.SCIPY, scipy_options={"method": "highs"})
            # prob.solve(cp.GUROBI)
    else:
        print("no trees")
        return None

    return allocation.value

# construct demand samples according to LP with tree demand distribution


def optimize_fn_ours(df, n_estimators, allocation_max, product):
    demand = np.array([df[f'demand{i}'] for i in range(n_estimators)]).T
    print(demand.shape)
    if np.shape(demand)[0] != 0:
        p = np.apply_along_axis(st.norm.fit, axis=1, arr=demand)
        rng = np.random.default_rng(10)
        demandN = []
        var = df['standardD']
        var = np.array(var)
        for i in range(len(p)):
            if var[i] > 80:
                d = st.norm.rvs(p[i, 0], var[i],
                                size=n_estimators, random_state=rng)
                demandN.append(d)
            else:
                d = st.norm.rvs(
                    p[i, 0], 80, size=n_estimators, random_state=rng)
                demandN.append(d)

        demand_mean = np.mean(demand, axis=1)
        demand_mean = np.array([demand_mean for _ in range(n_estimators)]).T
        demand = demand_mean + (demand - demand_mean) * 1.0
        demandN = np.array(demandN)
    else:
        demandN = demand
    return optimize_lp(demandN, allocation_max, product)


# evaluate unmet demand
def evaluate(df):
    return np.sum(np.maximum(df['target'] - df['allocation'], 0.0))


def get_stockouts(allocation, target):
    alloc = allocation[allocation['target'] > allocation['allocation']]
    return set(alloc['fac_id'])


def get_allocation_all(df, n_estimators, allocation_max, optimize_fn):
    df_allocation = []
    df = df[df['date'] != '2019-02-01']
    df = df[pd.notnull(df['date'])]
    # op = pd.read_csv(f'tmp/EM_Op.csv')
    # pdList = [0]
    for date in sorted(df['date'].unique()):
        print(date)
        for product in range(8):
            df_cur = get_allocation(
                df, n_estimators, allocation_max, date, product, optimize_fn)
            print(date, product, evaluate(df_cur))
            df_allocation.append(df_cur)
    return pd.concat(df_allocation, ignore_index=True)


def parser():
    # set-up parsers/subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Get predictions
    predictions = subparsers.add_parser('get-allocations')
    predictions.add_argument('--d', '--date', type=str, nargs='+',
                             help='Months you want for predictions, in YYYY-MM-01 form.')
    predictions.add_argument('--agg', '--aggregation', type=str,
                             help='aggregation level, e.g. district')
    predictions.add_argument('--fu', '--force-update', action='store_true',
                             help='Force update. If added, will overwrite existing files.')
    predictions.add_argument('--l', '--leadtime', type=int,
                             help='leadtime for forecasting. 1 means next month, 2 means two month ahead, etc.')
    return parser.parse_args()


def main():
    # args = pass in date, pass in allocation max(value), pass product(key)
    arg1 = sys.argv[1]
    arg2_path = sys.argv[2]
    if not os.path.isfile(arg2_path) or not arg2_path.lower().endswith('.xlsx'):
        print("Error: Argument 2 must be a valid XLSX file.")
        return
    # args = parser()
    # Step 0: Parameters
    # time1 = time.time()
    # want to add product for users to choose --> now is stored in config file
    # can we make dropdown to choose?
    allocation_max = 0  # can we let user to upload file? or no?
    n_estimators = 500
    dates = [arg1]  # ,'2022-02-01','2022-03-01'
    date_column = 'date'
    target_col = 'target'
    lead_time = 1
    # ('prop', optimize_fn_prop),('ours_dhis', optimize_fn_3mthAvg),('ours', optimize_fn_ours)
    approaches = [('ours', optimize_fn_ours)]
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    # Step 0.1: process dhis2 data
    # data = pd.read_csv("/Users/angelchung/Desktop/SL data & info/dfRaw_2023Q2July.csv") # add dhis2 pipeline to get this data
    # dhis2 = CleanDhis2(data,product_names)
    # BeforeFeature = ProcessBFeature(dhis2)
    # df4ML=create_features_essential_meds(BeforeFeature)
    # df4ML.to_csv("/Users/angelchung/Desktop/OptOldData2/tmp/df4ML8_2023.csv")

    # Step 1: Get predictions
    # train, test = get_predictions(dates, n_estimators, date_column, target_col, lead_time)
    # train = pd.read_csv(f'tmp/train500Jan2023.csv')
    # test = pd.read_csv(f'tmp/test500Jan2023NoNAtarget.csv')
    var = pd.read_csv(f'tmp/historicalVarJan2022.csv')
    # train = train.merge(var, on=['product', 'hf_pk'], how='left')
    # test = test.merge(var, on=['product', 'hf_pk'], how='left')

    # merge budget file to get allocation_max

    # if not os.path.isfile(arg2_path) or not arg2_path.lower().endswith('.xlsx'):
    # print("Error: Argument 2 must be a valid XLSX file.")
    # return
    op = pd.read_excel(arg2_path)
    # op = pd.read_csv(
    # "/Users/norrischen12/Desktop/dhis2Pipeline/tmp/BudgetConsump2.csv")
    # print("Hello!!!!")
    # op = op[op['dateF'] == '1/1/22'].copy()
    # train = train.merge(op, on=['product'], how='left')
    # test = test.merge(op, on=['product'], how='left')
    # print(test)

    # Step 2: Get weighted predictions
    # df_weight = get_allocation_all(train, n_estimators, allocation_max, optimize_fn_ours)
    # df_weight['weight'] = (df_weight['target'] > df_weight['allocation']).astype('float') + 0.05
    # df_weight['weight']=abs(df_weight['weight'])
    # df_weight = df_weight[['hf_pk', 'date', 'product','weight','unmetD','target','allocation']]
    # df_weight.to_csv(f'tmp/dfWeight_tree500Jan2023_WeightRaw.csv')
    # df_weight = pd.read_csv(f'tmp/dfWeight_tree500Jan2023_W005.csv', index_col=0)
    # df_weight = df_weight[['hf_pk', 'date', 'product', 'weight']]

    # train_weight, test_weight = get_predictions(dates, n_estimators, date_column, target_col, lead_time, df_weight)
    # test_weight.to_csv(f'tmp/testW_500Jan2023_W005.csv')
    test_weight = pd.read_csv(f'tmp/testW_500Jan2023_W005.csv')

    test_weight = test_weight.merge(op, on=['product'], how='left')
    test_weight = test_weight.merge(var, on=['product', 'hf_pk'], how='left')

    # Step 3: Optimization
    for name, optimize_fn in approaches:
        df_allocation = get_allocation_all(
            test_weight, n_estimators, allocation_max, optimize_fn)
        # df_allocation = pd.read_csv(
        # f'tmp/allocation_weighted_{name}_8_500tree0123_dhis2Com2_LPHisVarW005.csv')
        result = df_allocation[['product', 'hf_pk', 'allocation']]

        # df_allocation.to_csv(f'tmp/allocation_weighted_{name}_8_500tree0123_dhis2Com2_LPHisVarW005.csv')
        # print(result.to_json(orient='records'))
        if result.empty:
            print("Error: The result DataFrame is empty.")
        else:
            # Convert the DataFrame to an HTML table
            html_table = result.to_html(index=False)

        # Print the HTML table
            print(html_table)


if __name__ == "__main__":
    main()
