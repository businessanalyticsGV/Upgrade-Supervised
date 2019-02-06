#######################
# SUPERVISED MODELLING USING THE UNSUPERVISED DB
#######################

import time
start_time = time.time()

#I.- LOGISTIC REGRESSION AMONG OTHER TO MODEL PROBABILITIES

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns',500)

df = pd.read_csv('dataframe.csv')

### CREATED TARGET VARIABLE
df['upgrade'] = [1 if df.iloc[i][16] == 1 else 0 for i in range(df.shape[0])][1:]+[0]

print(df.shape)

###############################################
print('\n\nTime:'+str(time.time()-start_time))
###############################################