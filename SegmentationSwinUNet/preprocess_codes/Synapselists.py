# code to generate synapse lists from the synapse files in the synapse folder

import os

def generate_synapse_lists(synapse_dir, save_dir, save_file):
    filenames = os.listdir(synapse_dir)
    for filename in filenames:
            # print(filename)
            # input()
            # with open(os.path.join(save_dir, 'all.lst'), 'a') as file:
                # file.write(f'{filename}\n')
            
            # with open(os.path.join(save_dir, 'train.txt'), 'w') as file:
            # with open(os.path.join(save_dir, 'test_vol_full.txt'), 'a') as file:
            with open(os.path.join(save_dir, save_file), 'a') as file:
                file.write(f"{filename.split('.')[0]}\n")
                 

if __name__ == '__main__':
    # Directories
    synapse_dir = '../data/full_dataset/test_vol_h5_full'
    # synapse_dir = '../data/small_dataset/train_vol_h5_small'
    save_dir = '../lists/lists_Synapse'
    save_file = 'test_vol_full.txt'

    # Generate synapse lists
    generate_synapse_lists(synapse_dir, save_dir, save_file)
    print("Synapse lists saved")