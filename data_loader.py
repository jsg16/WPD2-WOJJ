# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 07:36:00 2022
@author: J Wang

Provide functions of loading data
"""

"""
    The following file structure is expected in the DATA_FOLDER:
    - DATA_FOLDER:
        - phase-1
            - {STATION} Combined Load xxxxx.csv
            - {STATION} Training Data.csv
            - template_1.csv
            - solution_phase1.csv (If XXX)
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

STATIONS = {"phase-1": ["BOURNVILLE CB 7", "BRADLEY STOKE CB 8", "STRATTON CB 4041"],
            "phase-2": []}

WEATHERS = {"BOURNVILLE CB 7": 7, "BRADLEY STOKE CB 8": 8, "STRATTON CB 4041": 3}

def load_data(DATA_FOLDER, phase, stations=[]):
    data_by_station = {}

    # default: all stations
    if stations is None or len(stations) == 0:
        stations = STATIONS["phase-%s" % phase]

    # map station name
    for station in stations:
        for s in STATIONS["phase-%s" % phase]:
            if s.startswith(station.upper()):
                data_by_station[s] = {"Weather Data": WEATHERS[s]}
                break

    # load data by station
    load_folder = os.path.join(DATA_FOLDER, "phase-%s" % phase)
    load_station_training_data(load_folder, data_by_station)

    # weather data by station
    weather_folder = os.path.join(DATA_FOLDER, "weather_data")
    load_weather(weather_folder, data_by_station)

    # national demand data
    nd_folder = os.path.join(DATA_FOLDER, "national_demand")
    national_demand = load_national_demand(nd_folder)

    return data_by_station, national_demand

def load_station_training_data(load_folder, data_by_station):
    for file in os.listdir(load_folder):
        if "Training Data" not in file:
            continue
        for station in data_by_station:
            if file.startswith(station.upper()):
                if "Training Data" in file:
                    data_by_station[station]["Training Data"] = os.path.join(load_folder, file)
                break
    for station, data in data_by_station.items():
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
