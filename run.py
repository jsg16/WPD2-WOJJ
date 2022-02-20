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
# Please refer to `data_loader.py` to ensure the structure inside the data folder.
DATA_FOLDER = os.path.join("..", "data")

# Specified the PHASE. Different phases contain different stations
PHASE = 1
# If PHASE is not set to 1 and 2, then you can specify the stations you want to run.
INPUT_STATIONS = ["BRADLEY STOKE CB 8", "HEMYOCK CB 56_24"] # examples

# Generate submission only works when PHASE is 1 or 2.
SUBMISSION_FLAG = True
# A folder for the output submission file if SUBMISSION_FLAG is True
SUBMISSION_PATH = os.path.join("..", "submissions")

# Smooth the input load. (True is recommended)
SMOOTH_INPUT = True
# Remove negative values in the prediction of EV chargers. (True is recommended)
APPLY_ABS = True
# Method for combined load smoothing
# Valid methods: 
#   - daily_max
#   - hourly_mean
#   - hourly_max
#   - averaged_smoothed_max
#   - weighted_smoothed_max
COMB_SMOOTH_METHOD = "averaged_smoothed_max"
# Window size for combined load smoothing
WS = 7

# GAM Parameters
# use N_SPLINES=4 to test functionality
# use N_SPLINES=10 to obtain the best result
N_SPLINES = 4
# coefficient of regularizer of GAM
LAMBDA = 0.1

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
    trained_gam = train_gam(dataset_by_station, return_fitted=False, return_test=True, gam_params={"lambda": LAMBDA}, te_params={"n_splines": N_SPLINES})

    # Step 5: Post-process the prediction
    generate_prediction(trained_gam, combined_load_by_station, method=COMB_SMOOTH_METHOD, ws=WS)

    # uncomment the lines below to show errors of phase-1 using different smoothing methods
    if SUBMISSION_FLAG and PHASE == 1:
        show_errors(trained_gam, combined_load_by_station, PHASE, DATA_FOLDER, apply_abs=APPLY_ABS)


    # uncomment the lines below to generate submission file
    if PHASE == 1 or PHASE == 2:
        generate_submission(trained_gam, PHASE, DATA_FOLDER, output_path=SUBMISSION_PATH, apply_abs=APPLY_ABS)
