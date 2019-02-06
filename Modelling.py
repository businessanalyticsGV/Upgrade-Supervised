#######################
# SUPERVISED MODELLING USING THE UNSUPERVISED DB
#######################

# I.- IMPORT DATASET WHICH IS LOCATED INSIDE A WINRAR FILE

import pandas as pd
import numpy as np

v_path = 'Z:/Modelos BA/1. 14Nov18 - Indexing/compiled_scored.gz'

df = pd.read_csv(v_path, compression = 'gzip')