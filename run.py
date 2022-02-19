# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 07:25:00 2022
@author: J Wang

Execution script for training and prediction
"""

"""
Workflow:

    Step 1: Load original data
    Step 2: Pre-process data
        - Weather: exponential smoothing with/without cubic transformation
        - National demand: weighted smoothing
        - Load: averaged smoothing and outlier removal
    Step 3: Prepare dataset
    Step 4: Train GAM
    Step 5: Post-process

"""

import os

from data_loader import load_data
from preprocess import *

import warnings
warnings.filterwarnings("ignore")

# Configurations
DATA_FOLDER = os.path.join("..", "data")
SMOOTH_INPUT = True


if __name__ == "__main__":
    
    # Step 1: Load original data
    data_by_station, national_demand = load_data(DATA_FOLDER, phase=1)

    # Step 2: Pre-process data
    # apply exponential smoothing to selected columns of weather data
    weather_smoothing(data_by_station)

    # apply weighted smoothing to national demands
    national_demand = weighted_smoothing(national_demand).value

    # apply outlier removal to loads of selected stations
    load_outlier_removal(data_by_station)

    # apply averaged smoothing to loads
    if SMOOTH_INPUT:
        load_smoothing(data_by_station)

    # Step 3: Prepare dataset
    dataset_by_station = pack_dataset(data_by_station, national_demand, input_smoothed=SMOOTH_INPUT)

    


