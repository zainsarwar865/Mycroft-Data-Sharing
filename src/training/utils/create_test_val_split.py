import os
import argparse
import hashlib
import configs
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--enola_base_dir', )   #/work/hdd/bdgs/npatil/test/Enola_Augmented/
parser.add_argument('--root_hash_config',)   #MT_food101_full_101
parser.add_argument('--original_dataset', type=str)  #food101
parser.add_argument('--original_config', type=str)   #full
parser.add_argument('--split_ratio', type=float)   #1
parser.add_argument('--seed',type=int)     #60
args = parser.parse_args()

assert args.split_ratio>0.0

root_config = args.root_hash_config
root_config_hash = (hashlib.md5(root_config.encode('UTF-8')))
mt_config = root_config + "_" + root_config_hash.hexdigest()    #MT_food101_full_101_somehexcode

mt_root_directory = os.path.join(args.enola_base_dir, mt_config)   #/work/hdd/bdgs/npatil/test/Enola_Augmented/MT_food101_full_101_somehexcode
datasets_folder = "Datasets"
datasets_dir = os.path.join(mt_root_directory, datasets_folder) #/work/hdd/bdgs/npatil/test/Enola_Augmented/MT_food101_full_101_somehexcode/Datasets

# Load val dataset
dataset_root = configs.dataset_root_paths[args.original_dataset] # /projects/bdgs/zsarwar/data/food-101/DF
val_dataset_config = configs.dataset_configs[args.original_dataset]['val'][args.original_config] #Loading val dataset pkl file from configs.py : df_food101_val.pkl
val_path = os.path.join(dataset_root, val_dataset_config) #/projects/bdgs/zsarwar/data/food-101/DF/df_food101_val.pkl
df_val = pd.read_pickle(val_path) 

unique_labels = df_val['label'].unique() #unique labels are extracted for stratified splitting

if args.split_ratio < 1:
    # Create two new DFs 
    df_val_hidden = None
    df_val_public = None
    split_ratio = args.split_ratio
    for uni_label in unique_labels:
        df_temp = df_val[df_val['label'] == uni_label]
        df_temp_public = df_temp.sample(frac=split_ratio, random_state=args.seed)
        df_temp_hidden = df_temp.drop(df_temp_public.index)

        if isinstance(df_val_hidden, pd.DataFrame):
            frames = [df_val_hidden, df_temp_hidden]
            df_val_hidden = pd.concat(frames)
        else:
            df_val_hidden = df_temp_hidden
            
        if isinstance(df_val_public, pd.DataFrame):
            frames = [df_val_public, df_temp_public]
            df_val_public = pd.concat(frames)
        else:
            df_val_public = df_temp_public

else:
    df_val_hidden = df_val
    df_val_public = df_val

    # Save both frames

val_public = val_dataset_config.split(".")[0] + "_val.pkl"
val_hidden = val_dataset_config.split(".")[0] + "_test.pkl"

val_public_path = os.path.join(datasets_dir, val_public)
val_hidden_path = os.path.join(datasets_dir, val_hidden)

df_val_hidden.to_pickle(val_public_path)
df_val_public.to_pickle(val_hidden_path)

print(f"Test-val split constructed at {val_public_path}" )

