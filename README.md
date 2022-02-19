# WPD2-WOJJ

This repository contains the codes and visualization tools for Western Power Distribution Data Challenge (Part 2). The aim of the challenge is to predict the peak EV usages across eight weeks in three substations in UK by using the demand and weather data only. You can find more details on the task can be found [here](https://codalab.lisn.upsaclay.fr/competitions/1324). The data for this task can be found [here](https://connecteddata.westernpower.co.uk/dataset/western-power-distribution-data-challenge-2-estimating-ev-charger-demand). YouTube kick-off can be found [here](https://www.youtube.com/watch?v=KMCmlDhpN8o). Our team **WOJJ** members include **W**angkun Xu, **O**layinka Ayo, **J**emima Graham, and **Jiaruijue** Wang. The work is supervised by Dr. Fei Teng. All the team members are from Control & Power group, Dept. EEE, Imperial College London, UK.

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

## Implementation 

To run the model on windows OS, we suggest to use Anaconda.

1. Install Anaconda;
2. Clone everything in this folder to a local folder, e.g. named as WPD2;
3. In the terminal run `conda create --name WPD2` to build the new environment;
4. Add the conda forge channel by `conda config --append channels conda-forge`;
5. Install the extra requirements `conda install --file requirements.txt`.