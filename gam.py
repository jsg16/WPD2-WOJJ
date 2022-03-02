# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 07:23:00 2022
@author: J Wang
"""

import pandas as pd
from pygam import GAM, te
from sklearn.metrics import mean_absolute_percentage_error

# CODE REVIEW (STYLE): These variables have inconsistent case (initial capitalisation) compared to others elsewhere in the project
Fixed_te_default = [['prev_2_mo', 'month', 'hour'],
                    ['month', 'hour', 'day'],
                    ['month', 'windspeed_north', 'windspeed_east']]

Nearby_te_default = [['prev_2_mo', 'national'], 
                     ['month', 'hour'], 
                     ['doW_x', 'doW_y']]

def initialise_gam(df, gam_params={"lam": 0.1}, te_params={"n_splines": 10}, num_fixed_col=12):
    # num_fixed_col depends on the feature
    # columns after num_fixed_col is the load of corresponding nearby stations
    col_index = {col: i for i, col in enumerate(df.columns)}
    tensor_terms = te(*[col_index[f] for f in Fixed_te_default[0]], **te_params)
    for features_comb in Fixed_te_default[1:-1]:
        tensor_terms += te(*[col_index[f] for f in features_comb], **te_params)
    # col starts from num_fixed_col stands for nearby stations
    for col in df.columns[num_fixed_col:]:
        te_flex = [[col] + features_comb for features_comb in Nearby_te_default[:-1]]
        for features_comb in te_flex:
            tensor_terms += te(*[col_index[f] for f in features_comb], **te_params)

    # If tensorterm is not in this order, the svd may fail to converge (don't know why)
    for features_comb in Fixed_te_default[-1:]:
        tensor_terms += te(*[col_index[f] for f in features_comb], **te_params)
    # col starts from num_fixed_col stands for nearby stations
    for col in df.columns[num_fixed_col:]:
        te_flex = [[col] + features_comb for features_comb in Nearby_te_default[-1:]]
        for features_comb in te_flex:
            tensor_terms += te(*[col_index[f] for f in features_comb], **te_params)

    if num_fixed_col == len(df.columns):
        te_flex = [['prev_2_mo', 'national'], ['national', 'doW_x', 'doW_y']]
        for features_comb in te_flex:
            tensor_terms += te(*[col_index[f] for f in features_comb], **te_params)
    gam = GAM(tensor_terms)
    gam.set_params(**gam_params)
    return gam

def train_gam(dataset_by_station, gam_params={"lam": 0.1}, te_params={"n_splines": 10},
              num_fixed_col=12, return_fitted=False, return_test=True):
    trained_gam = {}
    for station, dataset in dataset_by_station.items():
        train_df, test_df = dataset["train"], dataset["test"]
        train_df, train_label = train_df[test_df.columns], train_df["target"]
        gam = initialise_gam(train_df, gam_params=gam_params, te_params=te_params, num_fixed_col=num_fixed_col-1) # -1 due to the target column
        gam.fit(train_df, train_label)
        fitted = run_test(gam, train_df)
        train_mape = mean_absolute_percentage_error(train_label, fitted)
        print(station, ' Train MAPE: ', train_mape)
        trained_gam[station] = {"gam": gam, "train_mape": train_mape}
        if return_fitted:
            trained_gam[station]["fitted"] = fitted
        if return_test:
            trained_gam[station]["test_result"] = run_test(gam, test_df)
    return trained_gam

def run_test(gam, df):
    result = gam.predict(df)
    result = pd.Series(result, index = df.index)
    return result

