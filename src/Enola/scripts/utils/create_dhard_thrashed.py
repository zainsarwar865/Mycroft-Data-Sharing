import os
import argparse
import hashlib
import configs
import pandas as pd
import numpy as np
import configs

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--enola_base_dir', default='/bigstor/zsarwar/Enola_Augmented/')
parser.add_argument('--root_hash_config', type=str)
parser.add_argument('--mt_hash_config', type=str)
parser.add_argument('--original_dataset', type=str)
parser.add_argument('--original_config', type=str)
parser.add_argument('--trainer_type', type=str)
parser.add_argument('--bottom_k', type=int)
parser.add_argument('--per_class_budget', type=int)
parser.add_argument('--seed', type=int)


args = parser.parse_args()

root_config = args.root_hash_config
root_config_hash = (hashlib.md5(root_config.encode('UTF-8'))).hexdigest()
mt_config = root_config + "_" + root_config_hash

mt_root_directory = os.path.join(args.enola_base_dir, mt_config)
datasets_folder = "Datasets"
datasets_dir = os.path.join(mt_root_directory, datasets_folder)

# Read MT baseline folder
expr_config = args.mt_hash_config
expr_hash = (hashlib.md5(expr_config.encode('UTF-8'))).hexdigest()
expr_name = args.trainer_type + "_" + expr_hash
expr_dir = os.path.join(mt_root_directory, expr_name)
stats_path = os.path.join(expr_dir, "Metrics/agg_class_stats.pkl")
stats = pd.read_pickle(stats_path)
print(stats_path)

# Load the val set
original_dataset_path = configs.dataset_root_paths[args.original_dataset]

# Val needs to be the val_test.pkl set if training MT    
if args.trainer_type != "DO":    
    original_val_config = configs.dataset_configs[args.original_dataset]['val'][args.original_config]
    original_val_config = original_val_config.replace(".pkl", "_test.pkl")
    original_val_dataset_path = os.path.join(mt_root_directory, "Datasets")
    original_val_path = os.path.join(original_val_dataset_path, original_val_config)
    df_val = pd.read_pickle(original_val_path)


pred_masks = stats['prediction_masks']['best']
preds = stats['predictions']['best']
predicted_labels = stats['predicted_labels']['best']
uni_labels = df_val['label'].unique()
df_dhard = None
for lab in uni_labels:
    df_temp = df_val[df_val['label'] == lab]
    df_temp.insert(3, 'pred_label', predicted_labels[lab])

    # Sample incorrect ones
    df_dhard_temp = df_temp[~pred_masks[lab]]

    if isinstance(df_dhard, pd.DataFrame):
        frames = [df_dhard, df_dhard_temp]
        df_dhard = pd.concat(frames)
    else:
        df_dhard = df_dhard_temp


# subsample bottom k classes
"""
bottom_k = args.bottom_k
accs = list(stats['accuracy'].values())
accs = np.asarray(accs)
label_indices = np.argsort(accs)[0:bottom_k]
bottom_k_labels = np.asarray(list(stats['accuracy'].keys()))[label_indices]
"""
if args.original_dataset == 'food101':
        thrash_classes = ['French onion soup','Hot and sour soup','Pho','Takoyaki', 'Churros','Beignets', 'Lobster bisque', 'Sashimi', 'Clam chowder','Onion rings']
elif args.original_dataset == 'Imagenet':
    thrashed_classes = ['Welsh springer spaniel','Bedlington terrier','Pomeranian','pug, pug-dog','African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus','Bernese mountain dog','Boston bull, Boston terrier','Leonberg','Samoyed, Samoyede','borzoi, Russian wolfhound']
df_dhard_sub = df_dhard[df_dhard['class'].isin(thrashed_classes)]


# Create df_dhard_sub_aug by subsampling
aug_num = args.per_class_budget
seed = args.seed
seed = 121
dhard_labels = df_dhard_sub['label'].unique().tolist()
df_dhard_sub_aug = None

for lab in dhard_labels:
    df_dhard_temp = df_dhard_sub[df_dhard_sub['label'] == lab]
    if len(df_dhard_temp) < aug_num:
        print(f"Not enough samples to sample from for label {lab}")
        df_sampled = df_dhard_temp.sample(n=len(df_dhard_temp), random_state=seed)
    else:
        df_sampled = df_dhard_temp.sample(n=aug_num, random_state=seed)
    
    if isinstance(df_dhard_sub_aug, pd.DataFrame):
        frames = [df_dhard_sub_aug, df_sampled]
        df_dhard_sub_aug = pd.concat(frames)
    else:
        df_dhard_sub_aug = df_sampled

# Save dhard_empirical + dhard_sub
new_config = f"_empirical_{expr_hash}.pkl"
dhard_config = original_val_config.replace(".pkl", new_config)
dhard_path = os.path.join(original_val_dataset_path, dhard_config)
df_dhard_sub.to_pickle(dhard_path)
print("Dhard constructed, saved to %s" % dhard_path)

new_config = f"_sub_empirical_{expr_hash}.pkl"
dhard_config = original_val_config.replace(".pkl", new_config)
dhard_path = os.path.join(original_val_dataset_path, dhard_config)
df_dhard_sub_aug.to_pickle(dhard_path)

print("Dhard_sub constructed, saved to %s" % dhard_path)