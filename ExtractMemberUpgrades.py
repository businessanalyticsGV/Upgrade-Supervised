#######################
# SUPERVISED MODELLING USING THE UNSUPERVISED DB
#######################

import time
start_time = time.time()

# I.- IMPORT DATASET WHICH IS LOCATED INSIDE A GZIP FILE

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',500)

v_path = 'Z:/Modelos BA/1. 14Nov18 - Indexing/compiled_scored.gz'

df = pd.read_csv(v_path, compression = 'gzip')

print(df.shape)
print(df.head())

# II.- DETECTING MEMBERS WHO GOT AN UPGRADE

df_upgradedMembers = df[['%MemberID']][df['dst_LastContractAndToday'] == 1]
df = df_upgradedMembers.merge(df, how = 'left', on = ['%MemberID'])

print(df.shape)
print(df.head())
print('There are '+str(len(list(np.unique(df['%MemberID']))))+' members that upgraded')

df.to_csv('dataframe.csv',index = False)

###############################################
print('\n\nTime:'+str(time.time()-start_time))
###############################################