
# This script will read a stats.pkl file from the gradmatch run and the df_train set and save a retrieved DF which will then be fed to an MT_Augmented_D_Useful script to be merged with MT
# training data
# This needs to be a config file which I will run via another bash script to batch process all the retrieved samples
import os
import pandas as pd
import numpy as np
import pickle
import torch
from glob import glob
from tqdm import tqdm
import argparse
import yaml


parser = argparse.ArgumentParser()
# Args to reconstruct the hash function leading to the right directory for latent space data
############
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-config')
parser.add_argument('--per_class_gradmatch', type=bool, default=False)
parser.add_argument('--df_root', type=str, default="/bigstor/zsarwar/Imagenet_2012_subsets/MT_DO_Splits/")
parser.add_argument('--df_retrieve_config', type=str)
parser.add_argument('--gradmatch_retrieved_path', type=str, default="/bigstor/zsarwar/GradMatch")
parser.add_argument('--gradmatch_retrieved_config', type=str)
parser.add_argument('--gradmatch_retrieved_out_path', type=str, default="/bigstor/zsarwar/GradMatch/GradMatch_Retrievals/")
parser.add_argument('--tot_samples_per_class_needed', type=int, default=20)
parser.add_argument('--extra_classes_possible', type=bool, default=False)
parser.add_argument('--checkpoint_idx', type=int, default=1)




# Use config files

p = parser.parse_args()
if p.config is not None:
    with open(p.config, 'r') as f:
        default_arg = yaml.safe_load(f)
    key = vars(p).keys()
    for k in default_arg.keys():        
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    
    parser.set_defaults(**default_arg)

args = parser.parse_args()

ckpt_idx = args.checkpoint_idx
# Load stuff
do_config_path = os.path.join(args.df_root, args.df_retrieve_config)
df_do_train = pd.read_pickle(do_config_path)

# GradMatch retreived files
retrieved_config_path = os.path.join(args.gradmatch_retrieved_path, args.gradmatch_retrieved_config)
retrieved_stats_path = os.path.join(retrieved_config_path, "stats.pkl")
stats = pd.read_pickle(retrieved_stats_path)



# This will be ALL CLASSES based retrieval

d_hard_config = args.df_retrieve_config.replace("train", "test")
d_hard_path = os.path.join(args.df_root, d_hard_config)
df_mt_hard = pd.read_pickle(d_hard_path)
unique_classes_d_hard = df_mt_hard['class'].unique()
num_unique_classes_d_hard = len(unique_classes_d_hard)

all_weights = stats['all_classes']['encountered_weights_dict']
all_indices = stats['all_classes']['encountered_idxs_dict']
budget = args.tot_samples_per_class_needed * num_unique_classes_d_hard
# By default, use the second ckpt for now
idx_ckpt_to_use = list(stats['all_classes']['encountered_idxs_dict'].keys())[ckpt_idx]



# Construct a df_sub_strain which has all retrieved samples
# then weight by sorted weights and create final df_retrieved

df_subset = df_do_train.iloc[all_indices[idx_ckpt_to_use]]
argsort_indices = np.flip(np.argsort(all_weights[idx_ckpt_to_use].numpy()), axis=0)    
df_subset_weighted = df_subset.iloc[argsort_indices]
df_subset_weighted = df_subset_weighted[0:budget]
df_retrieved_gradmatch = df_subset_weighted.copy()

if not args.extra_classes_possible:
    # Remove all classes not presentg in D_hard
    df_retrieved_gradmatch = df_retrieved_gradmatch[df_retrieved_gradmatch['class'].isin(unique_classes_d_hard)]

# Save
out_df_name = "df_retrieved_" + args.gradmatch_retrieved_config + f"_retrieval-budget-perclass-{budget}.pkl"
full_outpath = os.path.join(args.gradmatch_retrieved_out_path, out_df_name)
df_retrieved_gradmatch.to_pickle(full_outpath)


