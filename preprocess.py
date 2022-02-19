# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 08:48:00 2022
@author: J Wang

Provide functions of pre-processing data
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

ALPHAS = {"temperature": 5e-2, "solar_irradiance": 5e-1, "windspeed_north": 5e-1, "windspeed_east": 5e-1}
OUTLIER_REMOVAL = {"BOURNVILLE CB 7"}

def avgeraged_smoothing(c, ws=7):
    weights = np.ones(ws) / ws
    conved = pd.DataFrame(data={"value":np.convolve(c.to_numpy(), weights, mode='same')}, index=c.index)
    return conved

def weighted_smoothing(c, ws=5):
    weights = np.arange(1, ws+1)
    weights[int(ws/2):] = weights[:int(ws/2)+1][::-1]
    weights = weights / weights.sum()
    return pd.DataFrame(data={"value":np.convolve(c.to_numpy(), weights, mode='same')}, index=c.index)

def exponential_smoothing(v, alpha, cubic=False):
    v_ = v if not cubic else (v + v**2 + v**3)
    return SimpleExpSmoothing(v_).fit(smoothing_level=alpha).fittedvalues

def weather_smoothing(data_by_station, weather_cols=ALPHAS.keys()):
    for station, data in data_by_station.items():
        data_by_station[station]["Weather Data"] = { col:
            exponential_smoothing(data["Weather Data"][col], ALPHAS[col], cubic=(col == "temperature"))
            for col in weather_cols
        }

def load_outlier_removal(data_by_station, n_times_std=3):
    for station, data in data_by_station.items():
        training_value = data["Training Data"].value
        if station in OUTLIER_REMOVAL:
            mean, std = np.mean(training_value), np.std(training_value)
            training_value[training_value > mean + n_times_std*std] = np.nan
            training_value[training_value < mean - n_times_std*std] = np.nan
        data_by_station[station]["Training Data"] = training_value

def load_smoothing(data_by_station, ws=7):
    for station, data in data_by_station.items():
        data["Training Data"] =  avgeraged_smoothing(data["Training Data"], ws=ws).value

NEARBY_STATIONS = {
    "BOURNVILLE CB 7": ["BRADLEY STOKE CB 8"], 
    "BRADLEY STOKE CB 8": ["BOURNVILLE CB 7"], 
    "STRATTON CB 4041": []
}

def pack_dataset(data_by_station, national_demand, input_smoothed=False):
    dataset_by_station = {station: {"train": None, "test": None} for station in data_by_station}
    for station, data in data_by_station.items():
        v = data["Training Data"]
        # train & test
        dfs = [pd.DataFrame(np.transpose([v.values[:-5376], v.values[5376:]]), index=v.index[5376:], columns=['prev_2_mo', 'target']),
               pd.DataFrame(np.transpose([v.values[-5376:]]), index=pd.date_range('2021-10-04', '2021-11-29', freq='15T', closed='left'), columns = ['prev_2_mo'])]
        for i, df in enumerate(dfs):
            # national demand
            df['national'] = national_demand
            # calendar features
            df['month'] = df.index.month
            df['hour'] = df.index.hour
            df['day'] =  df.index.dayofyear
            # temporal encoding of day of week
            df['doW_x'] =  np.sin(df.index.weekday / 7 * 2 * np.pi)
            df['doW_y'] =  np.cos(df.index.weekday / 7 * 2 * np.pi)
            # weather features
            for wf in ALPHAS.keys():
                df[wf] = data["Weather Data"][wf]
            # load of nearby station(s)
            for nearby_station in NEARBY_STATIONS[station]:
                if i == 0: # train
                    df[nearby_station] = data_by_station[nearby_station]["Training Data"].values[:-5376]
                else: # test
                    df[nearby_station] = data_by_station[nearby_station]["Training Data"].values[-5376:]
            # Drop nan value (including outliers) -> hourly data
            df.dropna(inplace=True)
        dataset_by_station[station]["train"] = dfs[0] if not input_smoothed else dfs[0].iloc[1:]
        dataset_by_station[station]["test"] = dfs[1]
    return dataset_by_station