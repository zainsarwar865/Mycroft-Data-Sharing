import os
import argparse
import hashlib

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--enola_base_dir')
parser.add_argument('--root_hash_config')

args = parser.parse_args()
root_config = args.root_hash_config
root_config_hash = (hashlib.md5(root_config.encode('UTF-8')))
mt_config = root_config + "_" + root_config_hash.hexdigest()

mt_root_directory = os.path.join(args.enola_base_dir, mt_config)
mt_baseline_dir = "MT_Baseline"
mt_augmented_dir = "MT_Augmented"
datasets_dir = "Datasets"
retrieval_dir = "Retrievals"
unicom_retrievals = "Unicom"
gradmatch_retrievals = "GradMatch"
joint_optim_retrievals = "JointOptimization"
random_retrievals = "Random"
full_retrievals = "Full"
metrics_dir = "Metrics"


# Create root
if not os.path.exists(mt_root_directory):
    os.makedirs(mt_root_directory)
else:
    print("MT's root directory already exists")
    #quit()

# Create subfolders in this directory for everything
retrieval_path = os.path.join(mt_root_directory, retrieval_dir)
unicom_path = os.path.join(mt_root_directory, unicom_retrievals)
gradmatch_path = os.path.join(mt_root_directory, gradmatch_retrievals)
random_path = os.path.join(mt_root_directory, random_retrievals)
full_path = os.path.join(mt_root_directory, full_retrievals)
joint_optim_path = os.path.join(mt_root_directory, joint_optim_retrievals)
# Create base folders
base_folders = [unicom_path,gradmatch_path,random_path,full_path, joint_optim_path]
for fold in base_folders:
    if not os.path.exists(fold):
        os.makedirs(fold)

# Create retrieval subfolders
retrieval_sub_dirs = [unicom_retrievals, gradmatch_retrievals, random_retrievals, full_retrievals, joint_optim_retrievals]
if not os.path.exists(retrieval_path):
    os.makedirs(retrieval_path)
    for sub_dir in retrieval_sub_dirs:
        t_dir = os.path.join(retrieval_path, sub_dir)
        os.makedirs(t_dir)

datasets_path = os.path.join(mt_root_directory, datasets_dir)
if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)

print("Created root directory")