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
df = df[df['mix'] == 'INT']

### CREATED TARGET VARIABLE
df['upgrade'] = [1 if df.iloc[i][16] == 1 else 0 for i in range(df.shape[0])][1:]+[0]

### SPLIT TEST AND TRAIN

ls_indexVariables = ['%MemberId','mix','score_date']
target = ['upgrade']
ls_contVariables = [c for c in df if c not in ls_indexVariables+target]

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(df[ls_contVariables],df[target],
test_size=.7, stratify=df[target])

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

dic_models = {'Logistic Regression':LogisticRegression().fit(X_train,y_train),
'Random Forest':RandomForestClassifier().fit(X_train,y_train),
'Neural Network Classifier': MLPClassifier().fit(X_train,y_train),
'Discriminant Analysis':QuadraticDiscriminantAnalysis().fit(X_train,y_train),
'KNeighbors Classifier':KNeighborsClassifier().fit(X_train,y_train)
# 'Suppor Vector Machine':SVC(kernel="linear", C=0.025).fit(X_train,y_train)}
}
### MODEL SCORING
for model in dic_models:
    print('\n\n++'+model)

    score = roc_auc_score(y_train,dic_models[model].predict(X_train))
    print('Train Model Score:'+str(score))

    score = roc_auc_score(y_test,dic_models[model].predict(X_test))
    print('Test Model Score:'+str(score))

###############################################
print('\n\n Time:'+str(time.time()-start_time))
###############################################