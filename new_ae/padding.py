import numpy as np
import pickle
import os

ori_bsu_dir = '/home/hgao53/alphafold_new/alphafold/final_bsu/'
save_dir = '/scratch/hgao53/af2_research_model/af2_output/'

bsu_file_names = os.listdir(ori_bsu_dir)
counter = 1

for bsu_name in bsu_file_names:
    print('\n')
    print('==============='+str(counter)+'===============')
    print('===============bsu===============')
    print(bsu_name)
    
    with open('/home/hgao53/alphafold_new/alphafold/final_bsu/' + bsu_name, 'rb') as f:
        data = pickle.load(f)

    ori = data['representations']['pair']
    print('===============ori_shape===============')
    print(ori.shape)

    padded = np.pad(ori, ((0, 2048-ori.shape[0]), (0, 2048-ori.shape[0]), (0, 0)), 'constant')
    print('===============padded_shape===============')
    print(padded.shape)
    
    for i in range(padded.shape[2]):
        tensor2d = padded[:, :, i]
        save_name = save_dir + bsu_name + '_' + str(i) + '.npy'
        if i == 1:
            print('===============saved_shape===============')
            print(tensor2d.shape)
        with open(save_name, 'wb') as f:
            pickle.dump(tensor2d, f)
            
    counter += 1