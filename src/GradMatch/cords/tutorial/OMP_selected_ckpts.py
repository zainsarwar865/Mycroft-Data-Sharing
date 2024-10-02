# TODOS
# need to extract weights at each training step
# resume from checkpoint thing
# get traceback form indices to raw training data
# Get back form 64x64 to 224x224 sized training images
# Add some other fancy features
import argparse
import yaml
import os
import sys
sys.path.append("/home/zsarwar/data_sharing/repo/Private-Data-Sharing/DL/GradMatch/cords")
os.chdir("/home/zsarwar/data_sharing/repo/Private-Data-Sharing/DL/GradMatch/cords")
import time
import numpy as np
import shutil
#================================================ SCRIPT TO DOS ================================================
"""
Add yaml based config and copy config to the output folder
Add checkpoint mid-point break reader - Done
Edit dictionary to store per-class results - Done
Edit dict to save after every epoch
Add functionality to switch test sets to val sets - No need to drop classes only need to switch the two datasets - Done
Edit get_dataset function to accept datasets as arguments - Done
"""
#===========================================================UTIL FUNCTIONS================================================================================================================================
def get_datasets(df_train, df_val, df_test):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

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

    val_total = 0
    tst_correct = 0
    tst_total = 0
    tst_loss = 0
    model.eval()
    logger_dict = {}
    if ("trn_loss" in print_args) or ("trn_acc" in print_args):
        samples=0
    
        with torch.no_grad():
            for _, data in enumerate(trainloader):
                inputs, targets = data

                inputs, targets = inputs.to(device), \
                                  targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                trn_loss += (loss.item() * trainloader.batch_size)
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
            for _, data in enumerate(valloader):
                inputs, targets = data
                inputs, targets = inputs.to(device), \
                                  targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += (loss.item() * valloader.batch_size)
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
            for _, data in enumerate(testloader):
                inputs, targets = data

                inputs, targets = inputs.to(device), \
                                  targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tst_loss += (loss.item() * testloader.batch_size)
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

parser.add_argument("-config")
parser.add_argument("--gpu_id", type=str)
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--trn_batch_size", type=int)
parser.add_argument("--val_batch_size", type=int)
parser.add_argument("--tst_batch_size", type=int)
parser.add_argument("--model_train_batch_size", type=int)
parser.add_argument("--data_fraction", type=float)
parser.add_argument("--select_subset_every", type=int)
parser.add_argument("--per_class", type=bool)
parser.add_argument("--numclasses", type=int)
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--print_every", type=int)
parser.add_argument("--save_every", type=int)
parser.add_argument("--pretrained", type=bool)
parser.add_argument("--frozen", type=bool)
parser.add_argument("--arch", type=str)
parser.add_argument("--config_base_dir", type=str)
parser.add_argument("--do_train_config", type=str)
parser.add_argument("--pretrained_ckpt_directory", type=str)
parser.add_argument("--root_save_dir", type=str)

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

# Copy config file to ckpt_directory

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']= args.gpu_id


from dataset import CustomImageDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from cords.utils.config_utils import load_config_data
import os.path as osp
from cords.utils.data.data_utils import WeightedSubset

from tqdm import tqdm
from torchvision.models import ResNet50_Weights, resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from cords.utils.models import ResNet50, ResNet18
import pickle
import pandas as pd
from glob import glob
import hashlib
import torchvision.transforms as transforms
import logging
import os
import os.path as osp
import sys
from cords.utils.data.dataloader.SL.adaptive import GLISTERDataLoader, AdaptiveRandomDataLoader, \
    CRAIGDataLoader, GradMatchDataLoader, RandomDataLoader, MILODataLoader, StochasticGreedyDataLoader, \
    WeightedRandomDataLoader
from dotmap import DotMap

expr_name = args.do_train_config.replace("df_train_", "").replace(".pkl", "")
expr_name = expr_name + "_gradmatch"
config = f'{expr_name}_pretrained-{args.pretrained}_frozen-{args.frozen}_model-{args.arch}_numclasses-{args.numclasses}_select-subset-every-{args.select_subset_every}_fraction-{args.data_fraction}_batchsize-{args.trn_batch_size}_model-train-batchsize-{args.model_train_batch_size}_start-{args.start}_end-{args.end}_perclass-{args.per_class}'

config_hash = (hashlib.md5(config.encode('UTF-8')))
extended_expr_name = expr_name + "_" + f'fraction-{args.data_fraction}_perclass-{args.per_class}'
config_hash =  extended_expr_name + "_" + config_hash.hexdigest()

out_dir = os.path.join(args.root_save_dir, config_hash)


if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print(f"Created directory: {out_dir}")
else:
    print(f"Directory already exists: {out_dir}")
    exit()

out_config_file = extended_expr_name + ".yaml"
if args.config is not None:
    shutil.copyfile(args.config, os.path.join(out_dir,out_config_file))


# Make ckpt directory
#ckpt_dir = os.path.join(out_dir, 'checkpoints')
#if not os.path.exists(ckpt_dir):
    #os.mkdir(ckpt_dir)

stats_path = os.path.join(out_dir, 'stats.pkl')

#Results logging directory
results_dir = out_dir
logger = __get_logger(results_dir)


all_ckpts = glob(args.pretrained_ckpt_directory)
all_ckpts = [ckpt for ckpt in all_ckpts if '.yaml' not in ckpt]
new_list = []

# Read the iter numbers
all_ckpt_iters = []
for ckpt in all_ckpts:
    iter_num = ckpt.split("/")[-1].split("_")[1]
    all_ckpt_iters.append(iter_num)
all_ckpt_iters = list(set(all_ckpt_iters))

for i in range(10):
    for ckpt in all_ckpts:
        for iter_num in all_ckpt_iters:
            if f'/{str(i)}_{iter_num}_checkpoint.pth.tar' in ckpt:
                new_list.append(ckpt)
                break    
            
new_list.sort()
for new_ckpt in new_list:
    idx = all_ckpts.index(new_ckpt)
    all_ckpts.pop(idx)

all_ckpts.sort()
all_ckpts = new_list + all_ckpts
all_ckpts = np.asarray(all_ckpts)

all_ckpts = all_ckpts[[1, 10, 20, 30, 40, 50, 60, 70, 80, 90]]

# For loop per unique class in valid set
do_train_config = args.do_train_config
do_val_config = do_train_config.replace("train", "test")
do_test_config = do_train_config.replace("train", "val")

df_train = pd.read_pickle(os.path.join(args.config_base_dir, do_train_config))
df_val = pd.read_pickle(os.path.join(args.config_base_dir, do_val_config))
df_test = pd.read_pickle(os.path.join(args.config_base_dir, do_test_config))

all_stats = {}
unique_classes = df_val['class'].unique()
data_fraction_per_class = args.data_fraction / len(unique_classes)

if args.per_class:

    for unique_class in unique_classes:
        # Creating the per class df_val
        
        df_val_class = df_val[df_val['class'] == unique_class]

        trainset, validset, testset = get_datasets(df_train, df_val_class, df_test)
        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.trn_batch_size,
                                                shuffle=False, pin_memory=True)

        valloader = torch.utils.data.DataLoader(validset, batch_size=args.val_batch_size,
                                                shuffle=False, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.tst_batch_size,
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

        optimizer = optim.SGD(model.parameters(), lr=5e-2,
                                        momentum=0.9,
                                        weight_decay=5e-4,
                                        nesterov=True)
        #T_max is the maximum number of scheduler steps. Here we are using the number of epochs as the maximum number of scheduler steps.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=200) 
        selection_strategy = 'GradMatchPB'
        dss_args = dict(model=model,
                        loss=criterion_nored,
                        eta=0.01,
                        num_classes=args.numclasses,
                        num_epochs=100,
                        device='cuda',
                        type="GradMatchPB",
                        fraction=data_fraction_per_class, 
                        select_every=args.select_subset_every,
                        lam=0.5,
                        selection_type='PerBatch',
                        v1=True,
                        valid=True,
                        kappa=0,
                        eps=1e-100,
                        linear_layer=True,
                        optimizer='lazy',
                        if_convex=False)
        dss_args = DotMap(dss_args)

        dataloader = GradMatchDataLoader(trainloader, valloader, dss_args, logger, 
                                        batch_size=args.model_train_batch_size,
                                        shuffle=True,
                                        pin_memory=False)
        #Training Arguments
        #print_args = ["val_loss", "val_acc", "tst_loss", "tst_acc", "trn_loss", "trn_acc", "time"]
        print_args = ["tst_loss", "tst_acc", "time"]
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


        """
        ################################################# Training Loop #################################################
        """
        train_time = 0
        for epoch in tqdm(range(len(all_ckpts[args.start:args.end]))):
        #for epoch in tqdm(range(0, num_epochs+1)):
            # Evaluating the Model at Regular Intervals
            if epoch == 0:
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
                #inputs = inputs.to(device)
                #targets = targets.to(device, non_blocking=True)
                #weights = weights.to(device)
                #outputs = model(inputs)
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
            stats['original_idxs'] = original_idxs
            stats['encountered_idxs'] = encountered_idxs
            stats['rem_idxs'] = rem_idxs
            stats['encountered_percentage'] = encountered_percentage
            stats['encountered_idxs_dict'] = dataloader.selected_idxs
            stats['encountered_weights_dict'] = dataloader.selected_weights
            stats['subset_weights'] = dataloader.subset_weights
            all_stats[unique_class] = stats
            with open(stats_path, 'wb') as of:
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

        """
        ################################################# Final Results Logging #################################################
        """

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

        # Save used indices, weights, final indices, weights etc in a dict of dict
        stats = {}
        original_idxs = set([x for x in range(len(trainset))])
        encountered_idxs = []
        for key in dataloader.selected_idxs.keys():
            encountered_idxs.extend(dataloader.selected_idxs[key])
        encountered_idxs = set(encountered_idxs)
        rem_idxs = original_idxs.difference(encountered_idxs)
        encountered_percentage = len(encountered_idxs) / len(original_idxs)
        stats['original_idxs'] = original_idxs
        stats['encountered_idxs'] = encountered_idxs
        stats['rem_idxs'] = rem_idxs
        stats['encountered_percentage'] = encountered_percentage
        stats['encountered_idxs_dict'] = dataloader.selected_idxs
        stats['encountered_weights_dict'] = dataloader.selected_weights
        stats['subset_weights'] = dataloader.subset_weights
        all_stats[unique_class] = stats


else:
    trainset, validset, testset = get_datasets(df_train, df_val, df_test)
    # Creating the Data Loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.trn_batch_size,
                                            shuffle=False, pin_memory=True)
    valloader = torch.utils.data.DataLoader(validset, batch_size=args.val_batch_size,
                                            shuffle=False, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.tst_batch_size,
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
    optimizer = optim.SGD(model.parameters(), lr=5e-2,
                                    momentum=0.9,
                                    weight_decay=5e-4,
                                    nesterov=True)
    #T_max is the maximum number of scheduler steps. Here we are using the number of epochs as the maximum number of scheduler steps.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=200) 
    selection_strategy = 'GradMatchPB'
    dss_args = dict(model=model,
                    loss=criterion_nored,
                    eta=0.01,
                    num_classes=args.numclasses,
                    num_epochs=100,
                    device='cuda',
                    type="GradMatchPB",
                    fraction=args.data_fraction, 
                    select_every=args.select_subset_every,
                    lam=0.5,
                    selection_type='PerBatch',
                    v1=True,
                    valid=True,
                    kappa=0,
                    eps=1e-100,
                    linear_layer=True,
                    optimizer='lazy',
                    if_convex=False)
    dss_args = DotMap(dss_args)

    dataloader = GradMatchDataLoader(trainloader, valloader, dss_args, logger, 
                                    batch_size=args.model_train_batch_size,
                                    shuffle=True,
                                    pin_memory=False)
    #Training Arguments
    #print_args = ["val_loss", "val_acc", "tst_loss", "tst_acc", "trn_loss", "trn_acc", "time"]
    print_args = ["tst_loss", "tst_acc", "time"]
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

    """
    ################################################# Training Loop #################################################
    """
    train_time = 0
    for epoch in tqdm(range(len(all_ckpts[args.start:args.end]))):
        # Evaluating the Model at Regular Intervals
        if epoch == 0:
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
            #inputs = inputs.to(device)
            #targets = targets.to(device, non_blocking=True)
            #weights = weights.to(device)
            #outputs = model(inputs)
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
        stats['original_idxs'] = original_idxs
        stats['encountered_idxs'] = encountered_idxs
        stats['rem_idxs'] = rem_idxs
        stats['encountered_percentage'] = encountered_percentage
        stats['encountered_idxs_dict'] = dataloader.selected_idxs
        stats['encountered_weights_dict'] = dataloader.selected_weights
        stats['subset_weights'] = dataloader.subset_weights
        all_stats['all_classes'] = stats
        with open(stats_path, 'wb') as of:
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

    """
    ################################################# Final Results Logging #################################################
    """

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


    # Save used indices, weights, final indices, weights etc in a dict of dict

    stats = {}

    original_idxs = set([x for x in range(len(trainset))])
    encountered_idxs = []

    for key in dataloader.selected_idxs.keys():
        encountered_idxs.extend(dataloader.selected_idxs[key])
    encountered_idxs = set(encountered_idxs)
    rem_idxs = original_idxs.difference(encountered_idxs)
    encountered_percentage = len(encountered_idxs)/len(original_idxs)

    stats['original_idxs'] = original_idxs
    stats['encountered_idxs'] = encountered_idxs
    stats['rem_idxs'] = rem_idxs
    stats['encountered_percentage'] = encountered_percentage
    stats['encountered_idxs_dict'] = dataloader.selected_idxs
    stats['encountered_weights_dict'] = dataloader.selected_weights
    stats['subset_weights'] = dataloader.subset_weights
    all_stats['all_classes'] = stats


with open(stats_path, 'wb') as of:
    pickle.dump(all_stats, of)





