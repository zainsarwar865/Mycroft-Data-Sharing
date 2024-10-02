import os
import pandas as pd
import numpy as np
import pickle
import torch
from glob import glob
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import yaml

parser = argparse.ArgumentParser(description="Analyzing OMP results")
parser.add_argument('-config')
parser.add_argument('-plot_dir', type=str, default="/bigstor/zsarwar/GradMatch/Plots/ckpt_ratios")
parser.add_argument('-plot_dir_metrics', type=str, default="/bigstor/zsarwar/GradMatch/Plots/ckpt_metrics")
parser.add_argument('-df_root', type=str, default="/bigstor/zsarwar/Imagenet_2012_subsets/MT_DO_Splits/")
parser.add_argument('-df_retrieve_config', type=str, default="")
parser.add_argument('-gradmatch_omp_root', type=str, default="/bigstor/zsarwar/GradMatch")
parser.add_argument('-gradmatch_omp_config', type=str, default="")
parser.add_argument('-per_class', type=bool, default=False)
parser.add_argument('--DO_type', type=str)

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
top_k_list = [25, 50, 100]
# Plot dir
plot_dir = args.plot_dir
plot_dir_metrics = args.plot_dir_metrics
plt.rcParams["figure.figsize"] = (18, 9)


# Load training and retrieved data
df_retrieve_path = os.path.join(args.df_root, args.df_retrieve_config)
df_train = pd.read_pickle(df_retrieve_path)
# Load the subset data
omp_retrieved_path = os.path.join(args.gradmatch_omp_root, args.gradmatch_omp_config)
omp_retrieved = os.path.join(omp_retrieved_path, "stats.pkl")

with open(omp_retrieved, 'rb') as i_file:
    stats = pickle.load(i_file)

# All Classes Analysis
if not args.per_class:
    
    ratios = []
    for k in stats['all_classes']['encountered_idxs_dict'].keys():
        best_indices = stats['all_classes']['encountered_idxs_dict'][k]
        df_subset = df_train.iloc[best_indices]
        rat  = [len(df_subset[df_subset['data_type'] == 'imagenet_baseline']), len(df_subset[df_subset['data_type'] == 'benign']), len(df_subset[df_subset['data_type'] == 'adversarial'])]
        rat = np.asarray(rat)
        rat = rat / np.sum(rat)
        rat = rat.tolist()
        ratios.append(rat)


    ckpts = list(stats['all_classes']['encountered_idxs_dict'].keys())
    ckpts = [2*val for val in ckpts]
    imagenet_ratio = [rat[0] for rat in ratios]
    dvw_benign_ratio = [rat[1] for rat in ratios]
    dvw_adv_ratio = [rat[2] for rat in ratios]

    val_losses = stats['all_classes']['metrics']['val_losses']

    val_accs = stats['all_classes']['metrics']['val_acc']


    # Plot for ratios
    df = {'Checkpoint': ckpts, 'Imagenet': imagenet_ratio, 'D v W - Benign': dvw_benign_ratio, 'D v W - Adversarial': dvw_adv_ratio}
    df = pd.DataFrame(df)
    df = df.set_index('Checkpoint')
    ax = sns.lineplot(data=df, dashes=False, legend='auto').set(title=f'OMP {args.DO_type} - All classes')
    filename = omp_retrieved.split("/")[-2] + '_unweighted' + '.jpg'
    outpath = os.path.join(plot_dir, filename)
    plt.savefig(outpath, dpi=300)
    plt.show()
    plt.clf()
    plt.figure()

    val_losses = stats['all_classes']['metrics']['val_losses']
    val_accs = stats['all_classes']['metrics']['val_acc']
    tst_losses = stats['all_classes']['metrics']['tst_losses']
    tst_accs = stats['all_classes']['metrics']['tst_acc']

    # Acc-loss plots
    df = {'Checkpoint': ckpts[1:], 'Val loss': val_losses, 'Val acc': val_accs, 'Test loss': tst_losses, 'Test acc': tst_accs}
    df = pd.DataFrame(df)
    df = df.set_index('Checkpoint')
    ax = sns.lineplot(data=df, dashes=False, legend='auto').set(title=f'OMP {args.DO_type} - All classes')
    filename = omp_retrieved.split("/")[-2] + '_unweighted_metrics' + '.jpg'
    outpath = os.path.join(plot_dir_metrics, filename)
    plt.savefig(outpath, dpi=300)


    # # Weight analysis

    all_weights = stats['all_classes']['encountered_weights_dict']
    all_indices = stats['all_classes']['encountered_idxs_dict']
    top_k = top_k_list
    for t_k in top_k:
        ratios = []
        for key in all_indices.keys():
            df_subset = df_train.iloc[all_indices[key]]
            argsort_indices = np.flip(np.argsort(all_weights[key].numpy()), axis=0)    
            df_subset_weighted = df_subset.iloc[argsort_indices]
            df_subset_weighted = df_subset_weighted[0:t_k]
            rat  = [len(df_subset_weighted[df_subset_weighted['data_type'] == 'imagenet_baseline']), len(df_subset_weighted[df_subset_weighted['data_type'] == 'benign']), len(df_subset_weighted[df_subset_weighted['data_type'] == 'adversarial'])]
            rat = np.asarray(rat)
            rat = rat / np.sum(rat)
            rat = rat.tolist()
            ratios.append(rat)
        ckpts = list(stats['all_classes']['encountered_idxs_dict'].keys())
        ckpts = [2*val for val in ckpts]
        imagenet_ratio = [rat[0] for rat in ratios]
        dvw_benign_ratio = [rat[1] for rat in ratios]
        dvw_adv_ratio = [rat[2] for rat in ratios]
        df = {'Checkpoint': ckpts, 'Imagenet': imagenet_ratio, 'D v W - Benign': dvw_benign_ratio, 'D v W - Adversarial': dvw_adv_ratio}
        df = pd.DataFrame(df)
        df = df.set_index('Checkpoint')
        plt.figure()
        plt.clf()
        sns.lineplot(data=df, dashes=False, legend='auto').set(title=f'OMP weighted Top-{t_k} {args.DO_type} - All classes')
        filename = omp_retrieved.split("/")[-2] + f'_weighted_top-k-{t_k}' + '.jpg'
        outpath = os.path.join(plot_dir, filename)
        plt.savefig(outpath, dpi=300)


        # Do Acc-loss plots
        plt.show()
        plt.clf()
        plt.figure()

        val_losses = stats['all_classes']['metrics']['val_losses']
        val_accs = stats['all_classes']['metrics']['val_acc']
        tst_losses = stats['all_classes']['metrics']['tst_losses']
        tst_accs = stats['all_classes']['metrics']['tst_acc']

        # Acc-loss plots
        df = {'Checkpoint': ckpts[1:], 'Val loss': val_losses, 'Val acc': val_accs, 'Test loss': tst_losses, 'Test acc': tst_accs}
        df = pd.DataFrame(df)
        df = df.set_index('Checkpoint')
        ax = sns.lineplot(data=df, dashes=False, legend='auto').set(title=f'OMP {args.DO_type} - All classes')
        filename = omp_retrieved.split("/")[-2] + f'_weighted_top-k-{t_k}_metrics' + '.jpg'
        outpath = os.path.join(plot_dir_metrics, filename)
        plt.savefig(outpath, dpi=300)

else:
    #Per Class Analysis 
    for label in stats.keys():
        ratios = []
        other_label = [lab for lab in list(stats.keys()) if lab != label][0]
        for k in stats[label]['encountered_idxs_dict'].keys():
            best_indices = stats[label]['encountered_idxs_dict'][k]
            
            df_subset = df_train.iloc[best_indices]
            rat  = [len(df_subset[df_subset['data_type'] == 'imagenet_baseline']), 
                    len(df_subset[(df_subset['data_type'] == 'benign') & (df_subset['class'] == label) ]),
                    len(df_subset[(df_subset['data_type'] == 'benign') & (df_subset['class'] == other_label) ]),
                    len(df_subset[(df_subset['data_type'] == 'adversarial') & (df_subset['class'] == label) ]),
                    len(df_subset[(df_subset['data_type'] == 'adversarial') & (df_subset['class'] == other_label) ])
                    ]
            rat = np.asarray(rat)
            rat = rat / np.sum(rat)
            rat = rat.tolist()
            ratios.append(rat)
        ckpts = list(range(1,len(ratios)+1))
        imagenet_ratio = [rat[0] for rat in ratios]
        dvw_benign_ratio_same = [rat[1] for rat in ratios]
        dvw_benign_ratio_opposite = [rat[2] for rat in ratios]
        dvw_adv_ratio_same = [rat[3] for rat in ratios]        
        dvw_adv_ratio_opposite = [rat[4] for rat in ratios]        
        df = {'Checkpoint': ckpts, 'Imagenet': imagenet_ratio, f'D v W - Benign-{label}': dvw_benign_ratio_same, f'D v W - Benign-{other_label}': dvw_benign_ratio_opposite, f'D v W - Adversarial-{label}': dvw_adv_ratio_same, f'D v W - Adversarial-{other_label}': dvw_adv_ratio_opposite}
        df = pd.DataFrame(df)
        df = df.set_index('Checkpoint')
        plt.figure()
        plt.clf()
        ax = sns.lineplot(data=df, dashes=False, legend='auto').set(title=f'OMP {args.DO_type} - {label}' )
        filename = omp_retrieved.split("/")[-2] + '_unweighted' + f'_class-{label}' '.jpg'
        outpath = os.path.join(plot_dir, filename)
        plt.savefig(outpath, dpi=300)


        # Do Acc-loss plots
        plt.show()
        plt.clf()
        plt.figure()

        val_losses = stats['all_classes']['metrics']['val_losses']
        val_accs = stats['all_classes']['metrics']['val_acc']
        tst_losses = stats['all_classes']['metrics']['tst_losses']
        tst_accs = stats['all_classes']['metrics']['tst_acc']

        # Acc-loss plots
        df = {'Checkpoint': ckpts[1:], 'Val loss': val_losses, 'Val acc': val_accs, 'Test loss': tst_losses, 'Test acc': tst_accs}
        df = pd.DataFrame(df)
        df = df.set_index('Checkpoint')
        ax = sns.lineplot(data=df, dashes=False, legend='auto').set(title=f'OMP {args.DO_type} - All classes')
        filename = omp_retrieved.split("/")[-2] + '_unweighted' + f'_class-{label}_metrics' '.jpg'
        outpath = os.path.join(plot_dir_metrics, filename)
        plt.savefig(outpath, dpi=300)

    # # Weight analysis

    for label in stats.keys():
        top_k = top_k_list
        all_weights = stats[label]['encountered_weights_dict']
        all_indices = stats[label]['encountered_idxs_dict']
        
        for t_k in top_k:
            ratios = []
            other_label = [lab for lab in list(stats.keys()) if lab != label][0]
            for key in all_indices.keys():
                df_subset = df_train.iloc[all_indices[key]]
                argsort_indices = np.flip(np.argsort(all_weights[key].numpy()), axis=0)    
                df_subset_weighted = df_subset.iloc[argsort_indices]
                df_subset_weighted = df_subset_weighted[0:t_k]
                rat  = [len(df_subset_weighted[df_subset_weighted['data_type'] == 'imagenet_baseline']), 
                        len(df_subset_weighted[(df_subset_weighted['data_type'] == 'benign') & (df_subset_weighted['class'] == label) ]),
                        len(df_subset_weighted[(df_subset_weighted['data_type'] == 'benign') & (df_subset_weighted['class'] == other_label) ]),
                        len(df_subset_weighted[(df_subset_weighted['data_type'] == 'adversarial') & (df_subset_weighted['class'] == label) ]),
                        len(df_subset_weighted[(df_subset_weighted['data_type'] == 'adversarial') & (df_subset_weighted['class'] == other_label) ])
                        ]
                        
                rat = np.asarray(rat)
                rat = rat / np.sum(rat)
                rat = rat.tolist()
                ratios.append(rat)

            ckpts = list(range(1,len(ratios)+1))
            imagenet_ratio = [rat[0] for rat in ratios]
            dvw_benign_ratio_same = [rat[1] for rat in ratios]
            dvw_benign_ratio_opposite = [rat[2] for rat in ratios]
            dvw_adv_ratio_same = [rat[3] for rat in ratios]        
            dvw_adv_ratio_opposite = [rat[4] for rat in ratios]        
            df = {'Checkpoint': ckpts, 'Imagenet': imagenet_ratio, f'D v W - Benign-{label}': dvw_benign_ratio_same, f'D v W - Benign-{other_label}': dvw_benign_ratio_opposite, f'D v W - Adversarial-{label}': dvw_adv_ratio_same, f'D v W - Adversarial-{other_label}': dvw_adv_ratio_opposite}
            df = pd.DataFrame(df)
            df = df.set_index('Checkpoint')
            plt.figure()
            plt.clf()
            ax = sns.lineplot(data=df, dashes=False, legend='auto').set(title=f'OMP weighted top-{t_k} {args.DO_type} - {label}')
            filename = omp_retrieved.split("/")[-2] + f'_weighted_top-k-{t_k}-class-{label}' + '.jpg'
            outpath = os.path.join(plot_dir, filename)
            plt.savefig(outpath, dpi=300)

print("Done")