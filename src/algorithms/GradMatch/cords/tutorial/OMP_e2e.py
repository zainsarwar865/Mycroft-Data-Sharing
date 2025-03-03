import argparse
import yaml
import os
import sys
#sys.path.append("/u/npatil/Cleanup/Mycroft-Data-Sharing/src/algorithms/GradMatch/cords")
#os.chdir("/u/npatil/Cleanup/Mycroft-Data-Sharing/src/algorithms/GradMatch/cords")
import time
import numpy as np
import hashlib

def get_datasets(df_train, df_val, df_test, arch):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    if arch == 'resnet50':
        train_dataset = CustomImageDataset(df_train, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = CustomImageDataset(df_val, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        test_dataset = CustomImageDataset(df_test, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.arch == 'efficientnet_b3':    
        
        train_dataset = CustomImageDataset(df_train, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(320, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = CustomImageDataset(df_val, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(320, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                normalize,
            ]))

        test_dataset = CustomImageDataset(df_test, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(320, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                normalize,
            ]))
        
    return train_dataset, val_dataset, test_dataset

def __get_logger(results_dir):
  os.makedirs(results_dir, exist_ok=True)
  # setup logger
  plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                      datefmt="%m/%d %H:%M:%S")
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  s_handler = logging.StreamHandler(stream=sys.stdout)
  s_handler.setFormatter(plain_formatter)
  s_handler.setLevel(logging.INFO)
  logger.addHandler(s_handler)
  f_handler = logging.FileHandler(os.path.join(results_dir, "results.log"))
  f_handler.setFormatter(plain_formatter)
  f_handler.setLevel(logging.INFO)
  logger.addHandler(f_handler)
  logger.propagate = False
  return logger

#TODO Edit this function
def load_ckpt(ckpt_path, model, optimizer):
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['loss']
    metrics = checkpoint['metrics']
    return start_epoch, model, optimizer, loss, metrics

def generate_cumulative_timing(mod_timing):
    tmp = 0
    mod_cum_timing = np.zeros(len(mod_timing))
    for i in range(len(mod_timing)):
        tmp += mod_timing[i]
        mod_cum_timing[i] = tmp
    return mod_cum_timing



def evaluate_model(curr_best_acc):
    """
    ################################################# Evaluation Loop #################################################
    """
    trn_loss = 0
    trn_correct = 0
    trn_total = 0
    val_loss = 0
    val_correct = 0
    val_total = 0
    tst_correct = 0
    tst_total = 0
    tst_loss = 0
    model.eval()
    logger_dict = {}

    if ("trn_loss" in print_args) or ("trn_acc" in print_args):
        samples=0
    
        with torch.no_grad():
            for _, data in enumerate(trainloader_eval):
                inputs, targets = data

                inputs, targets = inputs.to(device), \
                                  targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                trn_loss += (loss.item() * trainloader_eval.batch_size)
                samples += targets.shape[0]
                if "trn_acc" in print_args:
                    _, predicted = outputs.max(1)
                    trn_total += targets.size(0)
                    trn_correct += predicted.eq(targets).sum().item()
            trn_loss = trn_loss/samples
            trn_losses.append(trn_loss)
            logger_dict['trn_loss'] = trn_loss
        if "trn_acc" in print_args:
            trn_acc.append(trn_correct / trn_total)
            logger_dict['trn_acc'] = trn_correct / trn_total

    if ("val_loss" in print_args) or ("val_acc" in print_args):
        samples =0
        with torch.no_grad():
            for _, data in enumerate(valloader_eval):
                inputs, targets = data
                inputs, targets = inputs.to(device), \
                                  targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += (loss.item() * valloader_eval.batch_size)
                samples += targets.shape[0]
                if "val_acc" in print_args:
                    
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            val_loss = val_loss/samples
            val_losses.append(val_loss)
            logger_dict['val_loss'] = val_loss

        if "val_acc" in print_args:
            val_acc.append(val_correct / val_total)
            logger_dict['val_acc'] = val_correct / val_total

    if ("tst_loss" in print_args) or ("tst_acc" in print_args):
        samples =0
        with torch.no_grad():
            print("Eval testloader-eval")
            for _, data in tqdm(enumerate(testloader_eval)):
                inputs, targets = data

                inputs, targets = inputs.to(device), \
                                  targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tst_loss += (loss.item() * testloader_eval.batch_size)
                samples += targets.shape[0]
                if "tst_acc" in print_args:
                    _, predicted = outputs.max(1)
                    tst_total += targets.size(0)
                    tst_correct += predicted.eq(targets).sum().item()
            tst_loss = tst_loss/samples
            tst_losses.append(tst_loss)
            logger_dict['tst_loss'] = tst_loss

        if (tst_correct/tst_total) > curr_best_acc:
            curr_best_acc = (tst_correct/tst_total)

        if "tst_acc" in print_args:
            tst_acc.append(tst_correct / tst_total)
            best_acc.append(curr_best_acc)
            logger_dict['tst_acc'] = tst_correct / tst_total
            logger_dict['best_acc'] = curr_best_acc

    if "subtrn_acc" in print_args:
        if epoch == 0:
            subtrn_acc.append(0)
            logger_dict['subtrn_acc'] = 0
        else:    
            subtrn_acc.append(subtrn_correct / subtrn_total)
            logger_dict['subtrn_acc'] = subtrn_correct / subtrn_total

    if "subtrn_losses" in print_args:
        if epoch == 0:
            subtrn_losses.append(0)
            logger_dict['subtrn_loss'] = 0
        else: 
            subtrn_losses.append(subtrn_loss)
            logger_dict['subtrn_loss'] = subtrn_loss

    print_str = "Epoch: " + str(epoch)
    logger_dict['Epoch'] = epoch
    logger_dict['Time'] = train_time

    """
    ################################################# Results Printing #################################################
    """

    for arg in print_args:
        if arg == "val_loss":
            print_str += " , " + "Validation Loss: " + str(val_losses[-1])

        if arg == "val_acc":
            print_str += " , " + "Validation Accuracy: " + str(val_acc[-1])

        if arg == "tst_loss":
            print_str += " , " + "Test Loss: " + str(tst_losses[-1])

        if arg == "tst_acc":
            print_str += " , " + "Test Accuracy: " + str(tst_acc[-1])
            print_str += " , " + "Best Accuracy: " + str(best_acc[-1])

        if arg == "trn_loss":
            print_str += " , " + "Training Loss: " + str(trn_losses[-1])

        if arg == "trn_acc":
            print_str += " , " + "Training Accuracy: " + str(trn_acc[-1])

        if arg == "subtrn_loss":
            print_str += " , " + "Subset Loss: " + str(subtrn_losses[-1])

        if arg == "subtrn_acc":
            print_str += " , " + "Subset Accuracy: " + str(subtrn_acc[-1])

        if arg == "time":
            print_str += " , " + "Timing: " + str(timing[-1])

    logger.info(print_str)
#===========================================================UTIL FUNCTIONS================================================================================================================================

parser = argparse.ArgumentParser(description='OMP algorithm')
parser.add_argument("--gpu_id", type=str)
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--jump_ckpts", type=int)
parser.add_argument("--trn_batch_size", type=int)
parser.add_argument("--val_batch_size", type=int)
parser.add_argument("--tst_batch_size", type=int)
parser.add_argument("--model_eval_batch_size", type=int)
parser.add_argument("--num_candidates", type=int)
parser.add_argument("--select_subset_every", type=int)
parser.add_argument("--per_class", type=str)
parser.add_argument("--numclasses", type=int)
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--arch", type=str)
parser.add_argument("--joint_optimization", type=str)
# Enola args
parser.add_argument('--home_dir')
parser.add_argument('--enola_base_dir', default='/u/New/Enola_Augmented/')
parser.add_argument('--root_hash_config', type=str)
parser.add_argument('--mt_hash_config', type=str)
parser.add_argument('--do_hash_config', type=str)
parser.add_argument('--gradmatch_hash_config', type=str)
parser.add_argument('--DO_dataset', type=str)
parser.add_argument('--DO_config', type=str)
parser.add_argument('--MT_dataset', type=str)
parser.add_argument('--MT_config', type=str)
parser.add_argument('--per_class_budget', type=int)
parser.add_argument('--seed', type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']= args.gpu_id

# Deal with boolean variables
if args.per_class == "True":
    args.per_class = True
else:
    args.per_class = False


if args.joint_optimization == "True":
    args.joint_optimization = True
else:
    args.joint_optimization = False


root_config = args.root_hash_config
root_config_hash = (hashlib.md5(root_config.encode('UTF-8')))
args.root_hash_config = root_config_hash.hexdigest() 
mt_config = root_config + "_" + root_config_hash.hexdigest()
# Every file should be created inside this directory
mt_root_directory = os.path.join(args.enola_base_dir, mt_config)

do_config = args.do_hash_config
do_config_hash = (hashlib.md5(do_config.encode('UTF-8'))).hexdigest()
args.do_hash_config = do_config_hash

mt_hash_config = args.mt_hash_config
mt_hash_config = (hashlib.md5(mt_hash_config.encode('UTF-8'))).hexdigest()
args.mt_hash_config = mt_hash_config

do_config = do_config.split("_")[0] + "_" + do_config_hash
do_root_directory = os.path.join(mt_root_directory, do_config)
ckpts_folder = "Checkpoints"
# Use checkpoints from this directory
do_ckpt_dir = os.path.join(do_root_directory, ckpts_folder)

from dataset import CustomImageDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
#from cords.utils.config_utils import load_config_data
import os.path as osp
#from cords.utils.data.data_utils import WeightedSubset
from tqdm import tqdm
from torchvision.models import ResNet50_Weights, resnet50, ResNet50_Weights
cordspath = os.path.join(args.home_dir, 'algorithms', 'GradMatch', 'cords')
sys.path.append(cordspath)
os.chdir(cordspath)
from cords.utils.models import ResNet50
import pickle
import pandas as pd
from glob import glob
import hashlib
import torchvision.transforms as transforms
import logging
import os
import sys
from cords.utils.data.dataloader.SL.adaptive import GradMatchDataLoader 
from dotmap import DotMap
import torch.backends.cudnn as cudnn
import random
basepath = os.path.join(args.home_dir, 'training')
sys.path.append(basepath)
from utils import configs

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(args.seed)

gradmatch_config = args.gradmatch_hash_config
gradmatch_config_hash = hashlib.md5(gradmatch_config.encode('UTF-8')).hexdigest()
args.gradmatch_hash_config = gradmatch_config_hash 
expr_name = f"GradMatch_{gradmatch_config_hash}_{do_config_hash}"
gradmatch_retrievals_folder = "Retrievals/GradMatch/"
expr_path = os.path.join(mt_root_directory, gradmatch_retrievals_folder)
expr_dir = os.path.join(expr_path, expr_name)
if not os.path.exists(expr_dir):
    os.mkdir(expr_dir)

omp_results_path = os.path.join(expr_dir, f"OMP_{args.MT_dataset}_{args.DO_dataset}.pkl")
logger = __get_logger(expr_dir)

# Create and save YAML file
expr_config_dict = {}
all_args = args._get_kwargs()
expr_config_dict = {tup[0]:tup[1] for tup in all_args}
yaml_file = os.path.join(expr_dir, "Config.yaml")
with open(yaml_file, 'w') as yaml_out:
    yaml.dump(expr_config_dict, yaml_out)


all_ckpts = glob(do_ckpt_dir + "/*")
all_ckpts = [ckpt for ckpt in all_ckpts if 'best' not in ckpt]
new_list = []


# Read the iteration numbers
"""
all_ckpt_iters = []
for ckpt in all_ckpts:
    iter_num = ckpt.split("/")[-1].split("_")[1]
    all_ckpt_iters.append(iter_num)
all_ckpt_iters = list(set(all_ckpt_iters))
"""

# sort the first 10
for i in range(10):
    for ckpt in all_ckpts:
        if f'/{str(i)}_checkpoint.pth.tar' in ckpt:
            new_list.append(ckpt)
        
new_list.sort()
for new_ckpt in new_list:
    idx = all_ckpts.index(new_ckpt)
    all_ckpts.pop(idx)

all_ckpts.sort()
all_ckpts = new_list + all_ckpts

# Load DO and MT's datasets
do_dataset_path = configs.dataset_root_paths[args.DO_dataset]
# TODO REPLACE 'val' with 'train'
do_train_config = configs.dataset_configs[args.DO_dataset]['train'][args.DO_config]
do_train_path = os.path.join(do_dataset_path, do_train_config)

datasets_folder = "Datasets"
datasets_dir = os.path.join(mt_root_directory, datasets_folder)
val_config = configs.dataset_configs[args.MT_dataset]['val'][args.MT_config]
#new_config = f"_test_sub_empirical_{mt_hash_config}.pkl"    #THIS PATH WAS NOT CHANGED BY ME new path if error comes is commented from zains changes MAR 2025
new_config = f"_val_sub_empirical_{mt_hash_config}.pkl"   # THIS PATH IS CORRECT
dhard_config = val_config.replace(".pkl", new_config)
val_path = os.path.join(datasets_dir, dhard_config)
dhard_path = val_path
print("Dhard path is ", dhard_path)
test_config = configs.dataset_configs[args.MT_dataset]['val'][args.MT_config]
test_config = test_config.replace(".pkl", "_test.pkl")
test_path = os.path.join(datasets_dir, test_config)

df_train = pd.read_pickle(do_train_path)
df_val = pd.read_pickle(dhard_path)
df_test = pd.read_pickle(test_path)

all_stats = {}
unique_classes = df_val['class'].unique()
#data_budget_per_class = args.data_budget // len(unique_classes)

if args.per_class:

    for unique_class in unique_classes:
        # Creating the per class df_val    
        df_val_class = df_val[df_val['class'] == unique_class]
        df_train_class = df_train[df_train['class'] == unique_class]
        data_budget_per_class = args.num_candidates #* len(df_val_class)   # CHANGE DONE BY ZAIN MAR 2025
        data_budget_per_class = min(data_budget_per_class, len(df_train_class))

        logger.info(f"data_budget_per_class : {data_budget_per_class}")
        if len(df_train_class) == 0:
            logger.info(f"No samples for class : {unique_class} found in df_train")
            continue
            
        trainset, validset, testset = get_datasets(df_train_class, df_val_class, df_test, args.arch)
        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.trn_batch_size,
                                                shuffle=False, pin_memory=True)

        valloader = torch.utils.data.DataLoader(validset, batch_size=args.val_batch_size,
                                                shuffle=False, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.tst_batch_size,
                                                shuffle=False, pin_memory=True)
        
        trainloader_eval = torch.utils.data.DataLoader(trainset, batch_size=args.model_eval_batch_size,
                                        shuffle=False, pin_memory=True)
        
        testloader_eval = torch.utils.data.DataLoader(testset, batch_size=args.model_eval_batch_size,
                                                shuffle=False, pin_memory=True)

        valloader_eval = torch.utils.data.DataLoader(validset, batch_size=args.model_eval_batch_size,
                                            shuffle=False, pin_memory=True)
        
        device = 'cuda' 
        model = ResNet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(in_features=2048, out_features=args.numclasses)
        model = model.to(device)
        for params in model.parameters():
            params.requires_grad = False

        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        criterion = nn.CrossEntropyLoss()
        criterion_nored = nn.CrossEntropyLoss(reduction='none')

        """
        optimizer = optim.SGD(model.parameters(), lr=5e-2,
                                        momentum=0.9,
                                        weight_decay=5e-4,
                                        nesterov=True)
        
        #T_max is the maximum number of scheduler steps. Here we are using the number of epochs as the maximum number of scheduler steps.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=200) 
        """
        selection_strategy = 'GradMatchPB'
        dss_args = dict(model=model,
                        loss=criterion_nored,
                        eta=0.01,
                        num_classes=args.numclasses,
                        num_epochs=3,
                        device='cuda',
                        type="GradMatchPB",
                        fraction=data_budget_per_class, 
                        select_every=args.select_subset_every,
                        lam=0.5,
                        selection_type='PerBatch',
                        v1=True,
                        valid=True,
                        kappa=0,
                        eps=1e-100,
                        linear_layer=True,
                        optimizer='lazy',
                        joint_optimization=args.joint_optimization,
                        geometric_matrix_dict=None,
                        lam2=0,
                        curr_class=None,
                        if_convex=False)
        dss_args = DotMap(dss_args)        
        dataloader = GradMatchDataLoader(trainloader, valloader, dss_args, logger, 
                                        batch_size=args.model_eval_batch_size,
                                        shuffle=True,
                                        pin_memory=False)
        #Training Arguments
        #print_args = ["val_loss", "val_acc", "tst_loss", "tst_acc", "trn_loss", "trn_acc", "time"]
        #print_args = ["val_loss", "val_acc", "tst_loss", "tst_acc", "time"]
        print_args = ["val_loss", "val_acc", "time"]
        #Argumets for checkpointing
        #save_every = 10
        is_save = True
        #Evaluation Metrics
        trn_losses = list()
        val_losses = list()
        tst_losses = list()
        subtrn_losses = list()
        timing = [0]
        trn_acc = list()
        best_acc = list()
        curr_best_acc = 0
        val_acc = list()  
        tst_acc = list()  
        subtrn_acc = list()
        train_time = 0
        
        for epoch in tqdm(range(args.start, args.end, args.jump_ckpts)):
            # Evaluating the Model at Regular Intervals
            evaluate_model(curr_best_acc)
            subtrn_loss = 0
            subtrn_correct = 0
            subtrn_total = 0
            model.train()
            start_time = time.time()
            # Simulate training by loading checkpoints of DO's pretrained model
            loc = 'cuda:0'
            logger.info(f"Loading checkpoint {all_ckpts[epoch]}")
            checkpoint = torch.load(all_ckpts[epoch], map_location='cuda:0')
            model.load_state_dict(checkpoint['state_dict'])
            for _, (inputs, targets, weights) in enumerate(dataloader):
                pass
            epoch_time = time.time() - start_time
            timing.append(epoch_time)
            train_time += epoch_time

            # Update and save the dict at this point
            stats = {}
            original_idxs = set([x for x in range(len(trainset))])
            encountered_idxs = []

            for key in dataloader.selected_idxs.keys():
                encountered_idxs.extend(dataloader.selected_idxs[key])
            encountered_idxs = set(encountered_idxs)
            rem_idxs = original_idxs.difference(encountered_idxs)
            encountered_percentage = len(encountered_idxs) / len(original_idxs)
            
            # Metrics
            metrics = {
            #"trn_losses" :trn_losses,
            "val_losses" :val_losses,
            #"tst_losses" :tst_losses,
            "subtrn_losses":subtrn_losses,
            "timing":timing,
            "trn_acc":trn_acc,
            "best_acc":best_acc,
            "curr_best_acc":curr_best_acc ,
            "val_acc":val_acc  ,
            "tst_acc":tst_acc  ,
            "subtrn_acc":subtrn_acc,            
            }  
            
            stats['metrics'] = metrics
            stats['original_idxs'] = original_idxs
            stats['encountered_idxs'] = encountered_idxs
            stats['rem_idxs'] = rem_idxs
            stats['encountered_percentage'] = encountered_percentage
            stats['encountered_idxs_dict'] = dataloader.selected_idxs
            stats['encountered_weights_dict'] = dataloader.selected_weights
            stats['subset_weights'] = dataloader.subset_weights
            all_stats[unique_class] = stats
            with open(omp_results_path, 'wb') as of:
                pickle.dump(all_stats, of)

        ################################################# Results Summary #################################################

        original_idxs = set([x for x in range(len(trainset))])
        encountered_idxs = []
        # if self.cfg.dss_args.type != 'Full':
        for key in dataloader.selected_idxs.keys():
            encountered_idxs.extend(dataloader.selected_idxs[key])
        encountered_idxs = set(encountered_idxs)
        rem_idxs = original_idxs.difference(encountered_idxs)
        encountered_percentage = len(encountered_idxs)/len(original_idxs)

        logger.info("Selected Indices: ") 
        logger.info(dataloader.selected_idxs)
        logger.info("Percentages of data samples encountered during training: %.2f", encountered_percentage)
        logger.info("Not Selected Indices: ")
        logger.info(rem_idxs)                
        logger.info("GradMatchPB Selection Run---------------------------------")
        logger.info("Final SubsetTrn: {0:f}".format(subtrn_loss))
        if "val_loss" in print_args:
            if "val_acc" in print_args:
                logger.info("Validation Loss: %.2f , Validation Accuracy: %.2f", val_losses[-1], val_acc[-1])
            else:
                logger.info("Validation Loss: %.2f", val_losses[-1])

        if "tst_loss" in print_args:
            if "tst_acc" in print_args:
                logger.info("Test Loss: %.2f, Test Accuracy: %.2f, Best Accuracy: %.2f", tst_losses[-1], tst_acc[-1], best_acc[-1])
            else:
                logger.info("Test Data Loss: %f", tst_losses[-1])
        logger.info('---------------------------------------------------------------------')
        logger.info("GradMatchPB")
        logger.info('---------------------------------------------------------------------')

   
        ################################################# Final Results Logging #################################################
   

        if "val_acc" in print_args:
            val_str = "Validation Accuracy: "
            for val in val_acc:
                if val_str == "Validation Accuracy: ":
                    val_str = val_str + str(val)
                else:
                    val_str = val_str + " , " + str(val)
            logger.info(val_str)

        if "tst_acc" in print_args:
            tst_str = "Test Accuracy: "
            for tst in tst_acc:
                if tst_str == "Test Accuracy: ":
                    tst_str = tst_str + str(tst)
                else:
                    tst_str = tst_str + " , " + str(tst)
            logger.info(tst_str)

            tst_str = "Best Accuracy: "
            for tst in best_acc:
                if tst_str == "Best Accuracy: ":
                    tst_str = tst_str + str(tst)
                else:
                    tst_str = tst_str + " , " + str(tst)
            logger.info(tst_str)

        if "time" in print_args:
            time_str = "Time: "
            for t in timing:
                if time_str == "Time: ":
                    time_str = time_str + str(t)
                else:
                    time_str = time_str + " , " + str(t)
            logger.info(time_str)

        omp_timing = np.array(timing)
        omp_cum_timing = list(generate_cumulative_timing(omp_timing))
        logger.info("Total time taken by %s = %.4f ", "GradMatchPB", omp_cum_timing[-1])
