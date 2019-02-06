#######################
# SUPERVISED MODELLING USING THE UNSUPERVISED DB
#######################

import time
start_time = time.time()

#I.- LOGISTIC REGRESSION AMONG OTHER TO MODEL PROBABILITIES

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('dataframe.csv')

print(df.shape)
print(df.head())

###############################################
print('\n\nTime:'+str(time.time()-start_time))
###############################################