import numpy as np
import pickle
import os

bsu_file_names = os.listdir('/home/hgao53/alphafold_new/alphafold/final_bsu/')
counter = 1

for bsu_name in bsu_file_names:
    save_dir = '/home/hgao53/af2_research_model/bsu_padded/'
    print('\n')
    print('==============='+str(counter)+'===============')
    print('===============bsu===============')
    print(bsu_name)
    with open('/home/hgao53/alphafold_new/alphafold/final_bsu' + bsu_name, 'rb') as f:
        data = pickle.load(f)

    ori = data['representations']['pair']
    print('===============ori_shape===============')
    print(ori.shape)

    padded = np.pad(ori, ((0, 2048-ori.shape[0]), (0, 2048-ori.shape[0]), (0, 0)), 'constant')
    print('===============padded_shape===============')
    print(padded.shape)

    with open(save_dir+bsu_name, 'wb') as f:
        pickle.dump(padded, f)
    
    counter += 1