# WPD2-WOJJ

This repository contains the codes and visualization tools for Western Power Distribution Data Challenge (Part 2). The aim of the challenge is to predict the peak EV usages across eight weeks in three substations in UK by using the demand and weather data only. You can find more details on the task can be found [here](https://codalab.lisn.upsaclay.fr/competitions/1324). The data for this task can be found [here](https://connecteddata.westernpower.co.uk/dataset/western-power-distribution-data-challenge-2-estimating-ev-charger-demand). YouTube kick-off can be found [here](https://www.youtube.com/watch?v=KMCmlDhpN8o). Our team **WOJJ** members include **W**angkun Xu, **O**layinka Ayo, **J**emima Graham, and **J**iaruijue Wang. The work is supervised by Dr. Fei Teng. All the team members are from Control & Power group, Dept. EEE, Imperial College London, UK.

# Workflow

- Step 1: Load original data
- Step 2: Pre-process data
  - Weather: **exponential smoothing** with/without **cubic transformation**（smoothed by default)
  - National demand: **weighted smoothing** （smoothed by default)
  - Load: **averaged smoothing** and outlier removal (controlled by `SMOOTH_INPUT`)
- Step 3: Prepare dataset
  - Load data of 56 days ago (smoothed)
  - Smoothed national data
  - Calendar features: month, hour, day of year and temporal-encoded day of week
  - Weather features: ~~temperature~~, ~~solar_irradiance~~, windspeed_north and windspeed_east
  - Nearby-station load (smoothed)
- Step 4: Train GAM
  - **Combination of TensorTerms of different features**
- Step 5: Post-process
  - **Smoothed combined load** (controlled by `COMB_SMOOTH_METHOD` and window size `WS`)
    - daily max
    - hourly mean
    - hourly max
    - averaged smoothing + daily max
    - weighted smoothing + daily max
  - Negative value removal (controlled by `APPLY_ABS`)

# GAM

Generalized Additive Model, or GAM in short generalizes the linear models to capture different nonlinearities on each feature.

> GAMs relax the restriction that the relationship must be a simple weighted sum, and instead assume that the outcome can be modeled by a sum of arbitrary functions of each feature ([source](https://christophm.github.io/interpretable-ml-book/extend-lm.html#gam)).

GAM can be modelled as
$$
g(E_Y(y|x))=\beta_0+f_1(x_1)+f_2(x_2)+...+f_p(x_p)
$$
where $g(\cdot)$ is the link function of the mean of the distribution $P(Y|X)$. For each feature $x_i$, nonlinear transformation $f_i(\cdot)$ is applied：
$$
f_i(x_i)=\sum_{k=1}^{K_i}\beta_{ij}f_{ij}(x_i)
$$
$f_{ij}$ is the $k$-th basis function on feature $i$ and $K_i$ is the number of basis functions on feature $i$. Note that different features may have different types of basis functions.

## Pre-requisites 

To run the model on windows OS, we suggest to use Anaconda.

1. Install Anaconda;
2. Clone everything in this folder to a local folder, e.g. named as WPD2;
3. In the terminal run `conda create --name WPD2` to build the new environment;
4. Add the conda forge channel by `conda config --append channels conda-forge`;
5. Install the extra requirements `conda install --file requirements.txt`.

## Execution

To execute the codes, simply run the entry script `run.py`. 
 
```
python ./run.py
```

Before executing the entry script, you may need to modify the `DATA_FOLDER` and the `SUBMISSION_PATH` configurations in `run.py` and ensure the structure inside your data folder matches the description in the `data_loader.py`.

The following file structure is expected in the data_folder:
- data_folder:
    - phase-1
        - {STATION} Combined Load xxxxx.csv
        - {STATION} Training Data.csv
        - template_1.csv
        - solution_phase1.csv
    - phase-2
        - {STATION} Combined Load xxxxx.csv
        - {STATION} Training Data.csv
        - template_2.csv
    - weather_data
        - df_weather_{id}_hourly.csv
    - national_demand
        - demanddata_{year}.csv

We incorporate the following parameters for GAM:

* `N_SPLINES`: Number of splines to use for each marginal term. Must be of same length as feature.

* `LAMBDA`: Strength of smoothing penalty. Must be a positive float. Larger values enforce stronger smoothing.

## Error Matrix of stations of phase 1

With `SHOW_ERROR` set to True, various smoothing methods with different window size will be evaluated, which can guide the parameter selection in phase 2. Our best results (error matrix) over stations of phase 1 are summarized as followed, which is a `pandas.DataFrame` object.

|                    | daily_max | hourly_mean | hourly_max | avg-9 | **avg-13** | avg-17 | wgt-9 | wgt-13 | wgt-17 |
|--------------------|-----------|-------------|------------|-------------------------|--------------------------|--------------------------|-------------------------|--------------------------|--------------------------|
| BOURNVILLE CB 7    | 0.04363   | 0.04433     | 0.06339    | 0.04053                 | **0.03930**                  | 0.04052                  | 0.04081                 | 0.03996                  | 0.03945                  |
| BRADLEY STOKE CB 8 | 0.01887   | 0.01939     | 0.02598    | 0.01884                 | 0.01808                  | **0.01570**                  | 0.01853                 | 0.01878                  | 0.01793                  |
| STRATTON CB 4041   | 0.02663   | 0.03460     | 0.07314    | 0.02566                 | **0.02220**                  | 0.02744                  | 0.02465                 | 0.02517                  | 0.02341                  |
| Overall (mean)     | 0.02971   | 0.03277     | 0.05417    | 0.02834                 | **0.02653**                  | 0.02788                  | 0.02800                 | 0.02797                  | 0.02693                  |

*The avg and wgt stands for averaged and weighted smoothing respectively, while the followed number stands for the window size.*
