import os
import argparse
import hashlib
import configs
import pandas as pd
import numpy as np
import yaml
import configs


parser = argparse.ArgumentParser(description='Constructing Random Dhard')
parser.add_argument('--enola_base_dir')
parser.add_argument('--root_hash_config')
parser.add_argument('--do_hash_config', type=str)
parser.add_argument('--mt_hash_config', type=str)
parser.add_argument('--construct_hash_config', type=str,)
parser.add_argument('--MT_dataset', type=str)
parser.add_argument('--MT_config', type=str)
parser.add_argument('--DO_dataset', type=str)
parser.add_argument('--DO_config', type=str)
parser.add_argument("--num_candidates", type=int)
parser.add_argument('--per_class_budget', type=int)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

root_config = args.root_hash_config
root_config_hash = (hashlib.md5(root_config.encode('UTF-8')))
mt_config = root_config + "_" + root_config_hash.hexdigest()
mt_root_directory = os.path.join(args.enola_base_dir, mt_config)
args.root_hash_config = root_config_hash.hexdigest()

do_config = args.do_hash_config
do_config_hash = (hashlib.md5(do_config.encode('UTF-8')))
do_config_hash =  do_config_hash.hexdigest()

args.do_hash_config=do_config_hash

random_config = args.construct_hash_config
random_config_hash = hashlib.md5(random_config.encode('UTF-8')).hexdigest()
args.construct_hash_config = random_config_hash 
expr_name = f"Random_{random_config_hash}_{do_config_hash}"

random_retrievals_folder = "Retrievals/Random/"
expr_path = os.path.join(mt_root_directory, random_retrievals_folder)
expr_dir = os.path.join(expr_path, expr_name)
if not os.path.exists(expr_dir):
    os.mkdir(expr_dir)



mt_hash_config = args.mt_hash_config
mt_hash_config = (hashlib.md5(mt_hash_config.encode('UTF-8'))).hexdigest()
args.mt_hash_config = mt_hash_config

datasets_folder = "Datasets"
datasets_dir = os.path.join(mt_root_directory, datasets_folder)
val_config = configs.dataset_configs[args.MT_dataset]['val'][args.MT_config]

new_config = f"_test_sub_empirical_{mt_hash_config}.pkl"
dhard_config = val_config.replace(".pkl", new_config)

val_path = os.path.join(datasets_dir, dhard_config)
dhard_path = val_path
args.dhard_path = dhard_path
print("Dhard path is ", dhard_path)
# Load DHard
df_dhard = pd.read_pickle(dhard_path)



random_folder = "Random"
random_dir = os.path.join(mt_root_directory, random_folder)
random_retrievals_dir = os.path.join(mt_root_directory, random_retrievals_folder)
random_dataset_config =  f"df_{args.DO_dataset}_{args.DO_config}_{random_config_hash}_{random_config_hash}_{do_config_hash}.pkl"
random_dataset_path = os.path.join(random_dir, random_dataset_config)
random_retrievals_path = os.path.join(expr_dir, random_dataset_config)

# Create and save YAML file
expr_config_dict = {}
all_args = args._get_kwargs()
expr_config_dict = {tup[0]:tup[1] for tup in all_args}
yaml_file = os.path.join(expr_dir, "Config.yaml")
with open(yaml_file, 'w') as yaml_out:
    yaml.dump(expr_config_dict, yaml_out)

# Load DO_train dataset
dataset_root = configs.dataset_root_paths[args.DO_dataset]
train_dataset_config = configs.dataset_configs[args.DO_dataset]['train'][args.DO_config]
train_path = os.path.join(dataset_root, train_dataset_config)
df_train = pd.read_pickle(train_path)

## Load MT_train dataset
#dataset_root = configs.dataset_root_paths[args.MT_dataset]
#mt_train_dataset_config = configs.dataset_configs[args.MT_dataset]['train'][args.MT_config]
#mt_train_path = os.path.join(dataset_root, mt_train_dataset_config)
#df_mt_train = pd.read_pickle(mt_train_path)



do_train_classes = df_train['class'].unique()
unique_classes = df_dhard['class'].unique().tolist()
# Sample d_hard_budget samples from each class - classwise

df_useful = None

for uni_class in unique_classes:

    df_dhard_class = df_dhard[df_dhard['class'] == uni_class]
    budget = len(df_dhard_class) * args.num_candidates
    if uni_class not in do_train_classes:
        print(f"Label {uni_class} not found in df_train")
        continue 
    df_temp = df_train[df_train['class'] == uni_class]
    print(f"Class {uni_class} has {len(df_temp)} samples")
    df_useful_temp = df_temp.sample(n=budget, random_state=args.seed)

    if isinstance(df_useful, pd.DataFrame):
        frames = [df_useful, df_useful_temp]
        df_useful = pd.concat(frames)
    else:
        df_useful = df_useful_temp
        

df_useful.to_pickle(random_dataset_path)
df_useful.to_pickle(random_retrievals_path)
# Created random 
print(f"Created df_random_useful at {random_dataset_path}")