import numpy as np
import pandas as pd
import pickle
import re
import os
import math

# Set model name here, options = {'iMM904', 'iYO844', 'iML1515'}
model_name = 'iYO844'

model = pd.read_csv('../../raw/models/{}_Model.csv'.format(model_name))
if model_name == 'iMM904':
    model.rename(columns={'Kcat': 'kcat'}, inplace=True)
    model.rename(columns={'Km': 'km'}, inplace=True)
if model_name == 'iYO844':
    model.rename(columns={'BSU': 'Rule'}, inplace=True)
if model_name == 'iML1515':
    model.rename(columns={'meta_name': 'reac_meta_name'}, inplace=True)

print(model.shape)
# Drop useless columns
model = model.loc[:, ['Rule', 'reac_meta_name', 'reac_meta_value', 'Gibbs', 'kcat', 'km']]
model.head()