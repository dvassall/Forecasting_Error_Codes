import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import pickle
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score as R2
import matplotlib
import matplotlib.style as style

currDir = os.getcwd()

from sampleFuncs import *

# name of datafiles - individual files for 20, 100 m data
# timeFile = '_Hr'
# file20 = 'h20m'+timeFile+'.csv'
# file100 = 'h100m'+timeFile+'.csv'

# load necessary datafile from subfolder 'Data'
# data20 = pd.read_csv(os.path.join('Data',  file20), delimiter = ',')
# data100 = pd.read_csv(os.path.join('Data', file100), delimiter = ',')
# data20 = data20.iloc[:-1, :]

# lagged steps to include in inputs
n_steps = 2

# number of periods to skip for discrete timesteps
n_skip = 12 # 10 min -> 2, hour -> 12, 3hr -> 36

# Retrieve variables of interest
Data100 = Vars100m(data100, data20)
Data20 = Vars(data20)

# list variables of interest - complete list of variables
var = [['ws100'], ['dirNS', 'dirEW'], ['w'], ['t1', 't2'], ['Nsq10020'], ['dT10020'], ['T'], ['turbHtFlux'], ['richF10020'], ['richG10020'], ['rmsu'], ['fricVel'], ['tke'], ['TI'], ['normFricVel']]

# get variables of interest
trainIn, trainTar, testIn, testTar, varNames, trainTarWS, trainPred, testTarWS, testPred = data_Func(Data100, Data20, n_steps, n_skip, var)

# fix train/test target to get exogenous error
trainExo, testExo, errorBias = biasCorr(trainTar, testTar)

feats = RandomForestRegressor(n_estimators=1000,
                             max_features=0.7,
                             min_samples_split=100,
                             oob_score=True)
                             
feats.fit(trainIn, np.array(trainExo).flatten())

# Commented out section provides feature importance values
## Print out the feature and importances
#importances = (feats.feature_importances_*100).round(2)
#
## List of tuples with variable and importance
#feature_importances = [(feature, importance)
#                       for feature, importance in zip(trainIn.columns, importances)]
#
## Sort the feature importances by most important first
#feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#print('Oob Score:',feats.oob_score_.round(2), '\n')
#
#[print('Variable: {:20} Importance: {}%'.format(*pair)) for pair in feature_importances];

testErrPred = feats.predict(testIn)
real = np.array(testExo).flatten()

remainder = real-testErrPred
rmse = (np.sqrt(real**2)).mean()
newrmse = (np.sqrt(remainder**2)).mean()
r = np.round(R2(real, testErrPred)*100, 1)
