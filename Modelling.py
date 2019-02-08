#######################
# SUPERVISED MODELLING USING THE UNSUPERVISED DB
#######################

print('Modelling...')
import time
start_time = time.time()

#I.- LOGISTIC REGRESSION AMONG OTHER TO MODEL PROBABILITIES

import pandas as pd
import numpy as np

pd.set_option('display.max_columns',500)

df = pd.read_csv('dataframe.csv')
df = df[df['mix'] == 'INT']
df_newMembers = df.groupby(['%MemberID'], as_index = False)[['cnt_BroughtContracts']].max()
df_newMembers.rename(columns = {'cnt_BroughtContracts':'NewMember'},inplace=True)
df = df.merge(df_newMembers,how='left',on=['%MemberID'])
df = df[df['NewMember']!=1]

### CREATED TARGET VARIABLE

ls_compro = list(np.where(df['dst_LastContractAndToday'] == 1, 1,0)) 
df['compro'] = ls_compro[1:len(ls_compro)]+[0]

ls_membersBack = list(df['%MemberID'])
df['MemberAnterior'] = ls_membersBack[1:len(ls_membersBack)]+[0]

df['MemberAnterior'] = (df['MemberAnterior'] == df['%MemberID']).astype(int)

df['upgrade'] = [1 if ant+mem == 2 else 0 for ant,mem in zip(df['MemberAnterior'],df['compro'])]

df = df[[c for c in df if c not in ['MemberAnterior','compro','NewMember']]]
df.to_csv('test.csv',index=False)
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
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

X_train, X_test, y_train, y_test = train_test_split(df[ls_contVariables],df[target],
test_size=.7, stratify=df[target])

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

dic_models = {\
# 'Logistic Regression':LogisticRegression().fit(X_train,y_train), #y
'Random Forest':RandomForestClassifier().fit(X_train,y_train) #y
# 'Neural Network Classifier': MLPClassifier().fit(X_train,y_train) #y
# 'Discriminant Analysis':QuadraticDiscriminantAnalysis().fit(X_train,y_train),
# 'KNeighbors Classifier':KNeighborsClassifier().fit(X_train,y_train),
# 'Gaussian Naive Bayes':GaussianNB().fit(X_train,y_train) #y
# 'Gaussian Process Classifier':GaussianProcessClassifier().fit(X_train,y_train)
# 'Suppor Vector Machine':SVC().fit(X_train,y_train)
}

dic_params = {}
### MODEL SCORING
for model in dic_models:
    print('\n\n++'+model)

    score = roc_auc_score(y_train,dic_models[model].predict(X_train))
    print('Train Model Score:'+str(score))

    score = roc_auc_score(y_test,dic_models[model].predict(X_test))
    print('Test Model Score:'+str(score))

    dic_params[model] = dic_models[model].get_params()
    print('Parameters:\n',dic_params[model])

print(df.columns)
### GRIDSEARCH FOR PARAMETER OPTIMIZATION
# print('First is Done')

# from sklearn.model_selection import RandomizedSearchCV

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# rf = RandomForestClassifier()
# model_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
# random_state=42,n_jobs=-1)
    
# dic_models['RGS Random Trees'] = model_random.fit(X_train,y_train)

# dic_params = {}
# ### MODEL SCORING
# for model in dic_models:
#     print('\n\n++'+model)

#     score = roc_auc_score(y_train,dic_models[model].predict(X_train))
#     print('Train Model Score:'+str(score))

#     score = roc_auc_score(y_test,dic_models[model].predict(X_test))
#     print('Test Model Score:'+str(score))

#     dic_params[model] = dic_models[model].get_params()
#     print('Parameters:\n',dic_params[model])

###############################################
print('\n\n Time:'+str(time.time()-start_time))
###############################################