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

### SPLIT TEST AND TRAIN

ls_indexVariables = ['%MemberId','mix','score_date','scores']
target = ['upgrade']
ls_contVariables = [c for c in df if c not in ls_indexVariables+target]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(df[ls_contVariables],df[target],
test_size=.7, stratify=df[target])

model = LogisticRegression().fit(X_train,y_train)

### MODEL SCORING

print('Train Model Score:')
print(roc_auc_score(y_train,model.predict(X_train)))

print('Test Model Score:')
print(roc_auc_score(y_test,model.predict(X_test)))

###############################################
print('\n\n Time:'+str(time.time()-start_time))
###############################################