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

from data_loader import load_data, STATIONS
from preprocess import *
from gam import train_gam
from postprocess import generate_prediction, generate_submission, show_errors

import warnings
warnings.filterwarnings("ignore")

# Configurations
DATA_FOLDER = os.path.join("..", "data")
SUBMISSION_PATH = os.path.join("..", "submissions")
SMOOTH_INPUT = True
PHASE = 1
INPUT_STATIONS = ["BRADLEY STOKE CB 8", "HEMYOCK CB 56_24"] # specific stations work only if PHASE != 1, 2 or other customised phases

if __name__ == "__main__":

    print("PHASE %s" % PHASE)
    
    # Step 1: Load original data
    data_by_station, combined_load_by_station, national_demand = load_data(DATA_FOLDER)

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
    if PHASE == 1 or PHASE == 2:
        stations2run = STATIONS[(PHASE-1)*3:PHASE*3]
    else:
        stations2run = INPUT_STATIONS if INPUT_STATIONS is not None else []

    dataset_by_station = pack_dataset(data_by_station, national_demand, stations2run, input_smoothed=SMOOTH_INPUT)

    # Step 4: Traing GAM
    # use n_splines=4 to test functionality
    # use n_splines=10 to obtain the best result
    trained_gam = train_gam(dataset_by_station, return_fitted=False, return_test=True, te_params={"n_splines": 4})

    # Step 5: Post-process the prediction
    generate_prediction(trained_gam, combined_load_by_station)

    # uncomment the lines below to show errors of phase-1 using different smoothing methods
    if PHASE == 1:
        show_errors(trained_gam, combined_load_by_station, PHASE, DATA_FOLDER, apply_abs=True)


    # uncomment the lines below to generate submission file
    if PHASE == 1 or PHASE == 2:
        generate_submission(trained_gam, PHASE, DATA_FOLDER, output_path=SUBMISSION_PATH, apply_abs=True)
