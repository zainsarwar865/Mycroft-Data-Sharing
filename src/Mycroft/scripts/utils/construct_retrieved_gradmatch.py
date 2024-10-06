import os
import pandas as pd
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm
import argparse
import hashlib
import configs


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='GradMatch retrieval constructed')
# Enola args
parser.add_argument('--checkpoint_idx', type=int, default=1)
parser.add_argument('--enola_base_dir')
parser.add_argument('--root_hash_config', type=str)
parser.add_argument('--do_hash_config', type=str)
parser.add_argument('--gradmatch_hash_config', type=str)
parser.add_argument('--construct_hash_config', type=str)
parser.add_argument('--mt_hash_config', type=str)
parser.add_argument('--DO_dataset', type=str)
parser.add_argument('--DO_config', type=str)
parser.add_argument('--MT_dataset', type=str)
parser.add_argument('--MT_config', type=str)
parser.add_argument('--seed', type=int)

args = parser.parse_args()



# New stuff
#============================================================================
np.random.seed(args.seed)
# Get paths and construct hash configs

root_config = args.root_hash_config
root_config_hash = (hashlib.md5(root_config.encode('UTF-8')))
mt_config = root_config + "_" + root_config_hash.hexdigest()
# Every file should be created inside this directory
mt_root_directory = os.path.join(args.enola_base_dir, mt_config)
do_config = args.do_hash_config
do_config_hash = (hashlib.md5(do_config.encode('UTF-8'))).hexdigest()
#do_config_hash = do_config.split("_")[0] + "_" + do_config_hash

# Retrievals dir for GradMatch
gradmatch_retrievals_folder = "Retrievals/GradMatch"
gradmatch_retrievals_dir = os.path.join(mt_root_directory, gradmatch_retrievals_folder)
gradmatch_config = args.gradmatch_hash_config
gradmatch_config_hash = hashlib.md5(gradmatch_config.encode('UTF-8')).hexdigest()
expr_name = f"GradMatch_{gradmatch_config_hash}_{do_config_hash}"
expr_path = os.path.join(mt_root_directory, gradmatch_retrievals_folder)
expr_dir = os.path.join(expr_path, expr_name)

mt_hash_config = args.mt_hash_config
mt_hash_config = (hashlib.md5(mt_hash_config.encode('UTF-8'))).hexdigest()

# Augmented config
augmented_config = args.construct_hash_config
augmented_config_hash = hashlib.md5(augmented_config.encode('UTF-8')).hexdigest()

#============================================================================
# Load the DTrain
# DO's dataset path
do_dataset_path = configs.dataset_root_paths[args.DO_dataset]
#TODO change to train
do_train_config = configs.dataset_configs[args.DO_dataset]['train'][args.DO_config]
do_train_path = os.path.join(do_dataset_path, do_train_config)
df_train = pd.read_pickle(do_train_path)

# Load MT_Hard
datasets_folder = "Datasets"
datasets_dir = os.path.join(mt_root_directory, datasets_folder)
val_config = configs.dataset_configs[args.MT_dataset]['val'][args.MT_config]
#val_config = val_config.replace(".pkl", "_val.pkl")
new_config = f"_test_sub_empirical_{mt_hash_config}.pkl"
dhard_config = val_config.replace(".pkl", new_config)

dhard_path = os.path.join(datasets_dir, dhard_config)


df_dhard = pd.read_pickle(dhard_path)

# Total budget
unique_classes = df_dhard['class'].unique().tolist()
num_unique_classes_df_dhard = len(unique_classes)

ckpt_idx = args.checkpoint_idx
# GradMatch retreived files
retrieved_omp_path = os.path.join(expr_dir, f"OMP_{args.MT_dataset}_{args.DO_dataset}.pkl")
omp = pd.read_pickle(retrieved_omp_path)

"""
all_weights = omp['all_classes']['encountered_weights_dict']
all_indices = omp['all_classes']['encountered_idxs_dict']

# By default, use the second ckpt for now
idx_ckpt_to_use = list(omp['all_classes']['encountered_idxs_dict'].keys())[ckpt_idx]

# Construct a df_sub_strain which has all retrieved samples
# then weight by sorted weights and create final df_retrieved
df_subset = df_train.iloc[all_indices[idx_ckpt_to_use]]
argsort_indices = np.flip(np.argsort(all_weights[idx_ckpt_to_use].numpy()), axis=0)    
df_subset_weighted = df_subset.iloc[argsort_indices]
df_subset_weighted = df_subset_weighted[0:dhard_budget]
df_retrieved_gradmatch = df_subset_weighted.copy()

if not args.extra_classes_possible:
    # Remove all classes not presentg in D_hard
    df_retrieved_gradmatch = df_retrieved_gradmatch[df_retrieved_gradmatch['class'].isin(unique_classes)]
"""

df_retrieved_gradmatch = None

for uni_class in unique_classes:
    df_train_class = df_train[df_train['class'] == uni_class]
    if uni_class in omp:    
        all_weights = omp[uni_class]['encountered_weights_dict']
        all_indices = omp[uni_class]['encountered_idxs_dict']
        # By default, use the second ckpt for now
        idx_ckpt_to_use = list(omp[uni_class]['encountered_idxs_dict'].keys())[ckpt_idx]
        # Construct a df_sub_strain which has all retrieved samples
        # then weight by sorted weights and create final df_retrieved
        df_subset = df_train_class.iloc[all_indices[idx_ckpt_to_use]]
        argsort_indices = np.flip(np.argsort(all_weights[idx_ckpt_to_use].numpy()), axis=0)    
        df_subset_weighted = df_subset.iloc[argsort_indices]
        if isinstance(df_retrieved_gradmatch, pd.DataFrame):
            frames = [df_retrieved_gradmatch, df_subset_weighted]
            df_retrieved_gradmatch = pd.concat(frames)
        else:
            df_retrieved_gradmatch = df_subset_weighted
    else:
        print(f"Class {uni_class} does not exist in DO's dataset")

# Save retrieved gradmatch dataset
gradmatch_folder = "GradMatch"
gradmatch_dir = os.path.join(mt_root_directory, gradmatch_folder)
gradmatch_dataset_config =  f"df_{args.DO_dataset}_{args.DO_config}_{augmented_config_hash}_{gradmatch_config_hash}_{do_config_hash}.pkl"
gradmatch_dataset_path = os.path.join(gradmatch_dir, gradmatch_dataset_config)
df_retrieved_gradmatch.to_pickle(gradmatch_dataset_path)
print(f"Constructed GradMatch augmented dataset at {gradmatch_dataset_path}")