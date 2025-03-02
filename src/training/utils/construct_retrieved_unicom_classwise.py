import argparse
import os
import yaml
from glob import glob
from collections import defaultdict
import random
import shutil
import time
import warnings
import pickle
import numpy as np
import pandas as pd
from dataset import CustomImageDataset
import torch
import hashlib
import configs


parser = argparse.ArgumentParser()
# Args to reconstruct the hash function leading to the right directory for latent space data
############
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--seed', type=int, help="seed for pandas sampling", default=42)
#############
# New args
parser.add_argument('--num_candidates', type=int)
# Enola args
parser.add_argument('--enola_base_dir', default='/bigstor/zsarwar/Enola_Augmented/')
parser.add_argument('--root_hash_config', type=str)
parser.add_argument('--do_hash_config', type=str)
parser.add_argument('--unicom_hash_config', type=str)
parser.add_argument('--construct_hash_config', type=str)
parser.add_argument('--mt_hash_config', type=str)
parser.add_argument('--DO_dataset', type=str)
parser.add_argument('--DO_config', type=str)
parser.add_argument('--MT_dataset', type=str)
parser.add_argument('--MT_config', type=str)
parser.add_argument('--voting_scheme', type=str)



# Use config files
args = parser.parse_args()

np.random.seed(args.seed)
# Get paths and construct hash configs

root_config = args.root_hash_config
root_config_hash = (hashlib.md5(root_config.encode('UTF-8')))
mt_config = root_config + "_" + root_config_hash.hexdigest()
# Every file should be created inside this directory
mt_root_directory = os.path.join(args.enola_base_dir, mt_config)
do_config = args.do_hash_config
do_config_hash = (hashlib.md5(do_config.encode('UTF-8')))
do_config_hash = do_config_hash.hexdigest()

# Retrievals dir for unicom
unicom_retrievals_folder = "Retrievals/Unicom"
unicom_retrievals_dir = os.path.join(mt_root_directory, unicom_retrievals_folder)
unicom_config = args.unicom_hash_config
unicom_config_hash = hashlib.md5(unicom_config.encode('UTF-8')).hexdigest()
expr_name = f"Unicom_{unicom_config_hash}_{do_config_hash}"
expr_path = os.path.join(mt_root_directory, unicom_retrievals_folder)
expr_dir = os.path.join(expr_path, expr_name)

mt_hash_config = args.mt_hash_config
mt_hash_config = (hashlib.md5(mt_hash_config.encode('UTF-8'))).hexdigest()

datasets_folder = "Datasets"
datasets_dir = os.path.join(mt_root_directory, datasets_folder)
val_config = configs.dataset_configs[args.MT_dataset]['val'][args.MT_config]


new_config = f"_test_sub_empirical_{mt_hash_config}.pkl"
dhard_config = val_config.replace(".pkl", new_config)

dhard_path = os.path.join(datasets_dir, dhard_config)

df_dhard = pd.read_pickle(dhard_path)
dhard_classes = df_dhard['class'].unique().tolist()

val_config = val_config.replace(".pkl", "_test.pkl")
mt_val_path = os.path.join(datasets_dir, val_config)
df_mt_val = pd.read_pickle(mt_val_path)

# Augmented config
augmented_config = args.construct_hash_config
augmented_config_hash = hashlib.md5(augmented_config.encode('UTF-8')).hexdigest()

retrieved_config = f"{args.DO_dataset}_{args.DO_config}"
retrieved_config_path = os.path.join(expr_dir, retrieved_config)
# Read in top-k stuff
pred_index_path = retrieved_config_path + "_pred_index.t"
pred_label_path = retrieved_config_path + "_pred_label.t"
query_label_path = retrieved_config_path + "_query_label.t"

pred_index = torch.load(pred_index_path)
pred_label = torch.load(pred_label_path)
query_label = torch.load(query_label_path)

topk = args.num_candidates
#tot_required = args.d_hard_budget

# Load the DTrain
# DO's dataset path
do_dataset_path = configs.dataset_root_paths[args.DO_dataset]
do_train_config = configs.dataset_configs[args.DO_dataset]['train'][args.DO_config]
do_train_path = os.path.join(do_dataset_path, do_train_config)
df_train = pd.read_pickle(do_train_path)


if args.voting_scheme == 'borda-count':
    pass
    """
    # For loop for N classes of Borda Count
    all_dhard_classes = np.unique(query_label[0])
    all_candidates_votes = {}

    for unique_class in all_dhard_classes:
        class_indices = np.where(query_label[0] == unique_class)[0]
        class_votes = pred_index[0:topk, class_indices]

        # BORDA count implementation
        candidates_votes = {}
        all_candidates = np.unique(class_votes)
        for cand in all_candidates:
            candidates_votes[cand] = 0

        # Get weighted count of each candidate per vote
        tot_prefs = class_votes.shape[0]
        for pref_order in range(class_votes.shape[0]):
            for voter_id in range(class_votes.shape[1]):
                chosen_candidate = class_votes[pref_order, voter_id]
                weighted_votes = tot_prefs - 1 - pref_order
                candidates_votes[chosen_candidate]+= weighted_votes

        # Sort candidates
        counts = np.asarray(list(candidates_votes.values()))
        sorted_indices = np.flip(np.argsort(counts))
        # Make new dictionary with those values
        candidates_votes_sorted = {}
        curr_keys = list(candidates_votes.keys())
        for idx in sorted_indices:
            key = curr_keys[idx]
            candidates_votes_sorted[key] = candidates_votes[key]

        # Append dict
        all_candidates_votes[unique_class] = candidates_votes_sorted
    # While loop to select enough candidates to fill the budget
    # Remove all repeats (remove lowest ranked ones)
    all_candidates_classes = defaultdict(list)
    for unique_class in all_candidates_votes.keys():
        # Iterate over every candidate they voted for
        for i, cand in enumerate(all_candidates_votes[unique_class].keys()):
            all_candidates_classes[cand].append((unique_class, (i, cand)))  # (class, (ranking, candidate))

    # Remove bottom layered repeats from their respective dictionaries
    for cand in all_candidates_classes.keys():
        # Check if this candidate was repeated across classes
        if len(all_candidates_classes[cand]) > 1:
            # Yes
            # Sort by preference and pop all but one 
            all_class_voters = all_candidates_classes[cand].copy()
            all_class_voters = sorted(all_class_voters, key=lambda x: x[1][0])
            # Remove all starting from second position
            for voter in all_class_voters[1:]:
                all_candidates_votes[voter[0]].pop(voter[1][1])

    # Now select per_class_budget samples from each list and return them
    d_useful = []
    for unique_class in all_candidates_votes:
        all_rankings = list(all_candidates_votes[unique_class].keys())
        budgeted_rankings = all_rankings[0:tot_required]
        d_useful.append(budgeted_rankings)
            
    # Merge d_useful
    d_useful = [ele for class_budget in d_useful for ele in class_budget]
    print("Size of D_useful is : ",len(d_useful) )
    """

elif args.voting_scheme == 'Top_k_veto':

    df_retrieved = None

    unique_labels = np.unique(query_label[0])
    for lab in unique_labels:
        found_unique = False
        # Find columns for this label
        label_matches = np.where(query_label[0] == lab)[0]
        t_topk = topk
        budget = topk * len(label_matches)
        print("Budget is : ", budget)

        # Choose top_k
        while not found_unique:
        # Make submatrix
            sub_pred_index = pred_index[:, label_matches]
            sub_pred_index = sub_pred_index[0:t_topk, :]
            # Flatten and remove duplicates
            sub_pred_index = sub_pred_index.reshape(-1)
            sub_pred_index = np.unique(sub_pred_index)
            if len(sub_pred_index) == budget:
                found_unique = True
            else:
                print("Repeats found, increasing topk budget by 1")
                t_topk+=1
                if (t_topk - topk) > 20:
                    found_unique = True
            
        # Make df_sub
        df_train_sub = df_train[df_train['label'] == lab]
        df_train_sub = df_train_sub.iloc[sub_pred_index]
        if isinstance(df_retrieved, pd.DataFrame):
            frames = [df_retrieved, df_train_sub.copy(deep=True)]
            df_retrieved = pd.concat(frames)
        else:
            df_retrieved = df_train_sub.copy(deep=True)

# Save df_retrieved
unicom_folder = "Unicom"
unicom_dir = os.path.join(mt_root_directory, unicom_folder)
unicom_dataset_config =  f"df_{args.DO_dataset}_{args.DO_config}_{augmented_config_hash}_{unicom_config_hash}_{do_config_hash}.pkl"
unicom_dataset_path = os.path.join(unicom_dir, unicom_dataset_config)
df_retrieved.to_pickle(unicom_dataset_path)
print(f"Constructed unicom augmented dataset at {unicom_dataset_path}")