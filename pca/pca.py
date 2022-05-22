import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

AF2_OUTPUT_DIR = '/home/hgao53/alphafold_new/alphafold/final_bsu/'
SAVE_DIR = '/scratch/hgao53/pca/'


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
    # standardize
    scaler = StandardScaler()
    scaler.fit(logits_reshaped)
    logits_standard = scaler.transform(logits_reshaped)
    # PCA with n_components=1
    pca_2 = PCA(n_components=2)
    pca_2.fit(logits_standard)
    logits_PCA_2 = pca_2.transform(logits_standard)
    logits_after_PCA = logits_PCA_2.reshape(enzyme_len, enzyme_len, 2)

    # single representation
    # standardize
    scaler = StandardScaler()
    scaler.fit(single_repre)
    single_standard = scaler.transform(single_repre)
    # PCA with n_components=1
    pca_1 = PCA(n_components=1)
    pca_1.fit(single_standard)
    single_PCA_1 = pca_1.transform(single_standard)
    single_after_PCA = single_PCA_1.reshape(enzyme_len)

    # result
    result = dict()
    padded_logits = np.pad(logits_after_PCA, ((0, 1600-logits_after_PCA.shape[0]), (0, 1600-logits_after_PCA.shape[0]), (0, 0)), 'constant')
    padded_single = np.pad(single_after_PCA, ((0, 1600-single_after_PCA.shape[0])), 'constant')
    
    result['logits'] = padded_logits
    result['single'] = padded_single

    return result, enzyme_len


max_enzyme_len = 0
e_feature_list = dict()
for root, dirs, files in os.walk(AF2_OUTPUT_DIR):
    for file in files:
        if file.endswith(".pkl"):
            data = pickle.load(open(f'{AF2_OUTPUT_DIR}{file}', 'rb'))
            print(f'> processing {file}')
            result_dict, cur_enzyme_len = pca_main(data)
            
            e_name = file.split('.')[0]
            e_feature_list[e_name] = result_dict

            if cur_enzyme_len > max_enzyme_len:
                max_enzyme_len = cur_enzyme_len

            # with open(f'{SAVE_DIR}{file}', 'wb') as f:
            #     pickle.dump(result_dict, f)
            
with open(f'{SAVE_DIR}iYO844_feature.pkl', 'wb') as f:
    pickle.dump(e_feature_list, f)

print(f'Enzyme feature saved to {SAVE_DIR}iYO844_feature.pkl')
print(f'Number of enzymes: {len(e_feature_list)}')
print(f'Max enzyme length: {max_enzyme_len}')
print(f'Average enzyme length: {np.mean(list(e_feature_list.values())[0].values())}')
