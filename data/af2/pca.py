import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

AF2_OUTPUT_DIR = './'#'/home/alphafold_new/alphafold/final_bsu/'
SAVE_DIR = './'#'/scratch/hgao53/bsu_pca/'

def pca_main(data):
    # get single representations and logits
    representations = data['representations']
    single_repre = representations['single']

    distogram = data['distogram']
    logits_disto = distogram['logits']
    logits_reshaped = logits_disto.reshape(-1, 64)

    # the length of the enzyme
    enzyme_len = single_repre.shape[0]
    
    # logits
    ## standardize
    scaler = StandardScaler()
    scaler.fit(logits_reshaped)
    logits_standard = scaler.transform(logits_reshaped)
    ## PCA with n_components=1
    pca_1 = PCA(n_components=1)
    pca_1.fit(logits_standard)
    logits_PCA_1 = pca_1.transform(logits_standard)
    logits_after_PCA = logits_PCA_1.reshape(enzyme_len, enzyme_len)
    
    # single representation
    ## standardize
    scaler = StandardScaler()
    scaler.fit(single_repre)
    single_standard = scaler.transform(single_repre)
    ## PCA with n_components=1
    pca_1 = PCA(n_components=1)
    pca_1.fit(single_standard)
    single_PCA_1 = pca_1.transform(single_standard)
    single_after_PCA = single_PCA_1.reshape(enzyme_len)
    
    # result
    result = dict()
    result['logits'] = logits_after_PCA
    result['single'] = single_after_PCA
    
    return result, enzyme_len


max_enzyme_len = 0
for root, dirs, files in os.walk(AF2_OUTPUT_DIR):
    for file in files:
        if file.endswith(".pkl"):
            data = pickle.load(open(f'{AF2_OUTPUT_DIR}{file}', 'rb'))
            print(f'> processing {file}')
            result_dict, cur_enzyme_len = pca_main(data)
            
            if cur_enzyme_len > max_enzyme_len:
                max_enzyme_len = cur_enzyme_len
                
            with open(f'{SAVE_DIR}{file}', 'wb') as f:
                pickle.dump(result_dict, f)
                
print(f'max_enzyme_len: {max_enzyme_len}')
            