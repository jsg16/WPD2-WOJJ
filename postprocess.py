# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 08:44:00 2022
@author: J Wang

Provide functions of post-processing output of gam
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error

from data_loader import STATIONS
from preprocess import avgeraged_smoothing, weighted_smoothing

# functions of calculating daily max based on combined load and hourly prediction
# apply different methods on given combined loads to incorporate more information
def daily_max(c, p, ws=None):
    if ws is not None:
        print("window size not needed")
    return (c-p).resample('D').max()

def hourly_mean(c, p, ws=None):
    if ws is not None:
        print("window size not needed")
    return (c.resample('60T').mean()-p).resample('D').max()

def hourly_max(c, p, ws=None):
    if ws is not None:
        print("window size not needed")
    return (c.resample('60T').max()-p).resample('D').max()

def averaged_smoothed_max(c, p, ws):
    if ws is None or ws < 1:
        print("window size set to 13 by default")
        ws = 13
    return daily_max(avgeraged_smoothing(c, ws=ws).value, p)

def weighted_smoothed_max(c, p, ws):
    if ws is None or ws < 1:
        print("window size set to 13 by default")
        ws = 13
    return daily_max(weighted_smoothing(c, ws=ws).value, p)

# generate prediction by given smoothing methods and prediction
def generate_prediction(trained_gam, combined_load_by_station, method="averaged_smoothed_max", ws=None, save_key="prediction"):
    if method not in globals():
        print("Unknown post-process method `%s`, switching to default: `averaged_smoothed_max, ws=13`" % method)
        method, ws = "averaged_smoothed_max", 13
    for station, result in trained_gam.items():
        test_result = result["test_result"]
        trained_gam[station][save_key] = globals()[method](combined_load_by_station[station]["Combined Load"].value, test_result, ws=ws)
    
def generate_submission(trained_gam, PHASE, data_folder, output_path, apply_abs=True):
    stations = STATIONS[(PHASE-1)*3:PHASE*3]
    template = pd.read_csv(os.path.join(data_folder, "phase-%s" % PHASE, "template_%s.csv" % PHASE))
    template.iloc[:56, -1] = trained_gam[stations[0]]["prediction"]
    template.iloc[56:-56, -1] = trained_gam[stations[1]]["prediction"]
    template.iloc[-56:, -1] = trained_gam[stations[2]]["prediction"]
    if apply_abs:
        template.value[template.value < 0] = 0
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    output_file = "phase-%s_%s.csv" % (PHASE, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print("Submission csv dumped to:", os.path.join(output_path, output_file))
    template.to_csv(os.path.join(output_path, output_file))
    return template

"""
The output error matrix is a pandas.DataFrame object, with the structure:

                Method-1    Method-2    ...     Method-N  
station_1       MAPE        MAPE        ...     MAPE
station_2       MAPE        MAPE        ...     MAPE
station_3       MAPE        MAPE        ...     MAPE
overall         MAPE mean   MAPE mean   ...     MAPE mean

Best method and parameter can be retrieved by the errors over phase-1 stations.
"""
def show_errors(trained_gam, combined_load_by_station, PHASE, data_folder, apply_abs=True):
    stations = STATIONS[(PHASE-1)*3:PHASE*3]
    smoothing_candidate = [
        ("daily_max", None), ("hourly_mean", None), ("hourly_max", None), 
        ("averaged_smoothed_max", 9), ("averaged_smoothed_max", 13), ("averaged_smoothed_max", 17), 
        ("weighted_smoothed_max", 9), ("weighted_smoothed_max", 13), ("weighted_smoothed_max", 17)
    ]
    keys = []
    for (method, ws) in smoothing_candidate:
        keys.append(method if ws is None else "%s-%s" % (method, ws))
        generate_prediction(trained_gam, combined_load_by_station, method=method, ws=ws, save_key=keys[-1])
    solution = pd.read_csv(os.path.join(data_folder, "phase-%s" % PHASE, "solution_phase%s.csv" % PHASE))
    solution = [solution[:56], solution[56:-56], solution[-56:]]
    for s in solution:
        s.index = pd.to_datetime(s.date, format="%d/%m/%Y")
    errors = [[] for _ in range(len(stations))]
    for i, station in enumerate(stations):
        for key in keys:
            errors[i].append(mean_squared_error(solution[i].value.to_numpy(), 
                            np.abs(trained_gam[station][key].to_numpy()) if apply_abs else trained_gam[station][key].to_numpy()))
    errors = np.array(errors)
    errors = np.concatenate((errors, np.mean(errors, axis=0).reshape(1, -1)), axis=0)
    errors = pd.DataFrame(data=errors, columns=keys, index=stations+["overall"])
    print("="*20, "\nError Matrix of Phase\n")
    print(errors, "\n", "="*20)
    return errors
