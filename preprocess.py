# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 08:48:00 2022
@author: J Wang

Provide functions of pre-processing data
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# alpha for exponential smoothing
ALPHAS = {"temperature": 5e-2, "solar_irradiance": 5e-1, "windspeed_north": 5e-1, "windspeed_east": 5e-1}

# Only apply outlier removal to the following stations (based on experiments)
OUTLIER_REMOVAL = {"BOURNVILLE CB 7", "BRIDPORT CB 306", "PORTISHEAD ASHLANDS CB 4"}

NEARBY_STATIONS = {
    "BOURNVILLE CB 7": ["BRADLEY STOKE CB 8"], 
    "BRADLEY STOKE CB 8": ["BOURNVILLE CB 7"], 
    "STRATTON CB 4041": [],
    "BRIDPORT CB 306": ["HEMYOCK CB 56_24"], 
    "HEMYOCK CB 56_24": ["BRIDPORT CB 306"], 
    "PORTISHEAD ASHLANDS CB 4": ["BOURNVILLE CB 7", "BRADLEY STOKE CB 8"] # CODE REVIEW (DESIGN CHOICE): Why are Bournville and Bradley Stoke nearby to Portishead, but not vice-versa?
}

# CODE REVIEW (TYPO): Function name is mispelled
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

# Concatenate all the features here to create the dataset
def pack_dataset(data_by_station, national_demand, stations, input_smoothed=False):
    dataset_by_station = {station: {"train": None, "test": None} for station in stations}
    for station in stations:
        if station not in data_by_station:
            print("Invalid station name:", stations)
            continue
        data = data_by_station[station]
        v = data["Training Data"]
        # train & test
        # CODE REVIEW (STYLE): Repeated use of magic number (5376) that should be predefined (and explained). The same is probably 
        # true for the dates in the date range
        dfs = [pd.DataFrame(np.transpose([v.values[:-5376], v.values[5376:]]), index=v.index[5376:], columns=['prev_2_mo', 'target']),
               pd.DataFrame(np.transpose([v.values[-5376:]]), index=pd.date_range('2021-10-04', '2021-11-29', freq='15T', closed='left'), columns = ['prev_2_mo'])]
        for i, df in enumerate(dfs):
            # national demand
            df['national'] = national_demand
            # calendar features
            df['month'] = df.index.month
            # CODE REVIEW (DESIGN CHOICE): Why use temporal encoding for weekday but not hour?            
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