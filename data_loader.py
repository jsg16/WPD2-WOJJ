# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 07:36:00 2022
@author: J Wang

Provide functions of loading data
"""

"""
    The following file structure is expected in the data_folder:
    - data_folder:
        - phase-1
            - {STATION} Combined Load xxxxx.csv
            - {STATION} Training Data.csv
            - template_1.csv
            - solution_phase1.csv (If show errors)
        - phase-2
            - {STATION} Combined Load xxxxx.csv
            - {STATION} Training Data.csv
            - template_2.csv
        - weather_data
            - df_weather_{id}_hourly.csv
        - national_demand
            - demanddata_{year}.csv
"""

import os
import pandas as pd

STATIONS = ["BOURNVILLE CB 7", "BRADLEY STOKE CB 8", "STRATTON CB 4041", "BRIDPORT CB 306", "HEMYOCK CB 56_24", "PORTISHEAD ASHLANDS CB 4"]

WEATHERS = {"BOURNVILLE CB 7": 7, "BRADLEY STOKE CB 8": 8, "STRATTON CB 4041": 3,
            "BRIDPORT CB 306": 6, "HEMYOCK CB 56_24": 5, "PORTISHEAD ASHLANDS CB 4": 8}

def load_data(data_folder):
    data_by_station = {}
    combined_load_by_station = {}

    # map station name
    for station in STATIONS:
        data_by_station[station] = {"Weather Data": WEATHERS[station]}
        combined_load_by_station[station] = {}

    # load data by station
    load_station_training_data(os.path.join(data_folder, "phase-%s" % 1), data_by_station)
    load_station_training_data(os.path.join(data_folder, "phase-%s" % 2), data_by_station)
    # combined load by station
    load_station_combined_load(os.path.join(data_folder, "phase-%s" % 1), combined_load_by_station)
    load_station_combined_load(os.path.join(data_folder, "phase-%s" % 2), combined_load_by_station)

    # weather data by station
    weather_folder = os.path.join(data_folder, "weather_data")
    load_weather(weather_folder, data_by_station)

    # national demand data
    nd_folder = os.path.join(data_folder, "national_demand")
    national_demand = load_national_demand(nd_folder)

    return data_by_station, combined_load_by_station, national_demand

def load_station_training_data(load_folder, data_by_station):
    for file in os.listdir(load_folder):
        if "Training Data" not in file:
            continue
        for station, data in data_by_station.items():
            if file.startswith(station.upper()) and "Training Data" not in data:
                data_by_station[station]["Training Data"] = os.path.join(load_folder, file)
                break
    for station, data in data_by_station.items():
        if "Training Data" not in data or type(data["Training Data"]) is not str:
            continue
        # Training Data
        training = pd.read_csv(data["Training Data"])
        training.drop('Unnamed: 0', axis=1, inplace=True)
        # re-index
        training.index = pd.to_datetime(training.time)
        training.drop('time', axis=1, inplace=True)
        if all(training.units == 9):    # if all data has units = 9 (i.e. data in MW)  
            training.drop('units', axis=1, inplace=True)
        else:
            print(training.where(training.units!=9, inplace=True))
        data_by_station[station]["Training Data"] = training

def load_station_combined_load(load_folder, data_by_station):
    for file in os.listdir(load_folder):
        if "Combined Load" not in file:
            continue
        for station, data in data_by_station.items():
            if file.startswith(station.upper()) and "Combined Load" not in data:
                data_by_station[station]["Combined Load"] = os.path.join(load_folder, file)
                break
    for station, data in data_by_station.items():
        if "Combined Load" not in data or type(data["Combined Load"]) is not str:
            continue
        # Combined Load
        combined = pd.read_csv(data["Combined Load"])
        # re-index
        combined.index = pd.to_datetime(combined.time)
        combined.drop('time', axis=1, inplace=True)
        data_by_station[station]["Combined Load"] = combined

def load_weather(weather_folder, data_by_station):
    for station, data in data_by_station.items():
        weather_station = data["Weather Data"]
        weather = pd.read_csv(os.path.join(weather_folder, "df_weather_%s_hourly.csv" % weather_station))
        # re-index
        weather.index = pd.to_datetime(weather.datetime)
        weather.drop('datetime', axis=1, inplace=True)
        data_by_station[station]["Weather Data"] = weather

def load_national_demand(nd_folder, years=[2019, 2020, 2021]):
    national_demand = [pd.read_csv(os.path.join(nd_folder, "demanddata_%s.csv" % year)) for year in years]
    # re-index
    for i, nd in enumerate(national_demand):
        national_demand[i].index = pd.date_range('%s-01-01' % years[i], '%s-01-01' % (years[i]+1), freq='30T', closed='left')
    return pd.concat([nd.ENGLAND_WALES_DEMAND for nd in national_demand])
