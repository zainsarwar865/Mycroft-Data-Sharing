import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import yaml
import pickle
import numpy as np
import pandas as pd
from glob import glob
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights, resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Subset
from utils.dataset import CustomImageDataset
import torchvision.transforms as transforms
import hashlib
from sklearn.metrics import f1_score
import logging
from utils.configs import dataset_root_paths, dataset_configs
from collections import Counter
from utils.transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
from utils import utils

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--enola_base_dir')
parser.add_argument('--root_hash_config', type=str)
parser.add_argument('--mt_hash_config', type=str)
parser.add_argument('--do_hash_config', type=str)
parser.add_argument('--do_augmented_hash_config', type=str)
parser.add_argument('--do_augmented_construct_hash_config', type=str)
parser.add_argument('--augmentation_technique', type=str)
parser.add_argument('--arch', metavar='ARCH',)
parser.add_argument('--num_eval_epochs', type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--lr', dest='lr', type=float)
parser.add_argument('--lr_warmup_epochs', type=int)
parser.add_argument('--lr_warmup_decay',  type=float)
parser.add_argument('--lr_min', default=0.0, type=float)
parser.add_argument('--label_smoothing', type=float)
parser.add_argument("--mixup_alpha",  type=float)
parser.add_argument("--cutmix_alpha", type=float)
parser.add_argument("--auto_augment_policy", default='ta_wide', type=str)
parser.add_argument("--random_erasing", type=float)
parser.add_argument("--use_v2", default=False, type=bool)
parser.add_argument("--model_ema", type=bool)
parser.add_argument("--model_ema_steps",type=int,default=32)
parser.add_argument("--model_ema_decay",type=float, default=0.99998)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--weight_decay',  type=float)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--resume', type=str)
parser.add_argument('--evaluate', dest='evaluate', default=False)
parser.add_argument('--pretrained',type=str)
parser.add_argument('--world-size', default=-1, type=int)
parser.add_argument('--rank', default=-1, type=int)
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str)
parser.add_argument('--dist_backend', default='nccl', type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--multiprocessing_distributed', type=bool)
parser.add_argument('--seed', type=int, help="seed for pandas sampling")
parser.add_argument('--freeze_layers', type=str)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--external_augmentation', type=str)
parser.add_argument('--eval_pretrained', type=str)
parser.add_argument('--new_classifier', type=str)
parser.add_argument('--test_per_class', type=str)
parser.add_argument('--original_dataset', type=str)
parser.add_argument('--original_config', type=str)
parser.add_argument('--augmented_dataset', type=str)
parser.add_argument('--augmented_config', type=str)
parser.add_argument('--trainer_type', type=str)

"""
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
"""
args = parser.parse_args()

# Handle bash boolean variables
if args.pretrained == "True":
    args.pretrained = True
else:
    args.pretrained = False

if args.freeze_layers == "True":
    args.freeze_layers = True
else:
    args.freeze_layers = False

if args.external_augmentation == "True":
    args.external_augmentation = True
else:
    args.external_augmentation = False


if args.new_classifier == "True":
    args.new_classifier = True
else:
    args.new_classifier = False

if args.test_per_class == "True":
    args.test_per_class = True
else:
    args.test_per_class = False


if args.eval_pretrained == "True":
    args.eval_pretrained = True
else:
    args.eval_pretrained = False



if args.model_ema == "True":
    args.model_ema = True
else:
    args.model_ema = False

# assertions to avoid divide by 0 issue in learning rate
assert args.epochs>args.lr_warmup_epochs, "Total epochs must be more than warmup epochs"

best_acc1 = 0

root_config = args.root_hash_config
root_config_hash = (hashlib.md5(root_config.encode('UTF-8')))
mt_config = root_config + "_" + root_config_hash.hexdigest()

# Every file should be created inside this directory
mt_root_directory = os.path.join(args.enola_base_dir, mt_config)

# Check trainer type
if args.trainer_type == "MT_Baseline":
    #expr_config = f"{args.trainer_type}_{args.original_dataset}_{args.original_config}_pretrained-{args.pretrained}_{args.arch}_freeze-layers-{args.freeze_layers}_lr-{args.lr}_epochs-{args.epochs}_eval-pretrained-{args.eval_pretrained}"
    expr_config = args.mt_hash_config 
    
elif args.trainer_type == "MT_Augmented":
    #expr_config = f"{args.trainer_type}_{args.original_dataset}_{args.original_config}_{args.augmented_dataset}_{args.augmented_config}_{args.arch}_pretrained-{args.pretrained}_freeze-layers-{args.freeze_layers}_lr-{args.lr}_epochs-{args.epochs}__eval-pretrained-{args.eval_pretrained}"
    expr_config = f"{args.mt_hash_config}_{args.do_hash_config}_{args.do_augmented_construct_hash_config}_{args.do_augmented_hash_config}"
    ### Convert strings to hashes
    do_hash_config = (hashlib.md5(args.do_hash_config.encode('UTF-8')))
    do_hash_config =  do_hash_config.hexdigest()
    args.do_hash_config = do_hash_config

    do_augmented_hash_config = (hashlib.md5(args.do_augmented_hash_config.encode('UTF-8')))
    do_augmented_hash_config =  do_augmented_hash_config.hexdigest()
    args.do_augmented_hash_config = do_augmented_hash_config

    do_augmented_construct_hash_config = (hashlib.md5(args.do_augmented_construct_hash_config.encode('UTF-8')))
    do_augmented_construct_hash_config =  do_augmented_construct_hash_config.hexdigest()
    args.do_augmented_construct_hash_config = do_augmented_construct_hash_config
    ###
elif args.trainer_type == "DO": 
    # TODO - Revisit after DO details are known
    expr_config = args.do_hash_config

expr_hash = (hashlib.md5(expr_config.encode('UTF-8')))
expr_name = args.trainer_type + "_" + expr_hash.hexdigest()
expr_dir = os.path.join(mt_root_directory, expr_name)


if not os.path.exists(expr_dir):
    os.makedirs(expr_dir)

# Make checkpoints and metrics directory
ckpt_folder = "Checkpoints"
ckpt_dir = os.path.join(expr_dir, ckpt_folder)
metrics_folder = "Metrics"
metrics_dir = os.path.join(expr_dir, metrics_folder)

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)


# MT Augmented
if args.external_augmentation:        
        # DO_dataset now needs to be a string of hash configs
        augmented_config = f"df_{args.augmented_dataset}_{args.augmented_config}_{do_augmented_construct_hash_config}_{do_augmented_hash_config}_{do_hash_config}.pkl"
        augmented_df_dir = os.path.join(mt_root_directory, args.augmentation_technique)
        augmented_df_path = os.path.join(augmented_df_dir, augmented_config)


# Create log files
if(args.evaluate):
    logging_path = os.path.join(expr_dir,"test_log.log")
else:
    logging_path = os.path.join(expr_dir,"train_val_log.log")



logging.basicConfig(filename=logging_path,
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)



# Create and save YAML file
expr_config_dict = {}
all_args = args._get_kwargs()
expr_config_dict = {tup[0]:tup[1] for tup in all_args}
yaml_file = os.path.join(expr_dir, "Config.yaml")
with open(yaml_file, 'w') as yaml_out:
    yaml.dump(expr_config_dict, yaml_out)

def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    """
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])        
        
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        #mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    """
    if args.multiprocessing_distributed:

        ngpus_per_node = torch.cuda.device_count()
        mp.spawn(main_worker,  args=(ngpus_per_node,), nprocs=ngpus_per_node, join=True)
    else:
        if torch.cuda.is_available():
            ngpus_per_node = torch.cuda.device_count()
        main_worker(ngpus_per_node, args)

def main_worker(rank, ngpus_per_node):
    global best_acc1    
    setup(rank, ngpus_per_node)
    torch.cuda.set_device(rank)
    """
    #if args.distributed:
        #if args.dist_url == "env://" and args.rank == -1:
            #args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + rank
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=rank)
    """
    # Dataset paths

    original_dataset_path = dataset_root_paths[args.original_dataset]
    original_train_config = dataset_configs[args.original_dataset]['train'][args.original_config]

    retrieved_path = os.path.join(original_dataset_path, original_train_config)
    df_original_train = pd.read_pickle(retrieved_path)
    train_classes = df_original_train['label'].unique()
    args.num_classes = len(train_classes)
    # create model
    if args.pretrained:
        logger.critical(f"=> using pre-trained model {args.arch}")
        if args.arch == 'resnet50':
            model = models.__dict__[args.arch](weights=ResNet50_Weights.IMAGENET1K_V2)
        elif args.arch == 'efficientnet_b3':
            model = models.__dict__[args.arch](weights=EfficientNet_B3_Weights.DEFAULT)
        if args.new_classifier:
            if args.arch == 'resnet50':
               model.fc = nn.Linear(in_features=2048, out_features=args.num_classes, bias=True)
            elif args.arch == 'efficientnet_b3':
                model.classifier[1] = nn.Linear(in_features=1536, out_features=args.num_classes, bias=True)
    
    else:
        logger.critical(f"=> creating model {args.arch}")
        model = models.__dict__[args.arch]()
        
        if args.new_classifier:
            if args.arch == 'resnet50':
                model.fc = nn.Linear(in_features=2048, out_features=args.num_classes, bias=True)
            elif args.arch == 'efficientnet_b3':
                model.classifier[1] = nn.Linear(in_features=1536, out_features=args.num_classes, bias=True)


    
    
    
    # Add option to freeze/unfreeze more layers
    # TODO
    if args.freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        if args.arch == 'resnet50':
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        elif args.arch == 'efficientnet_b3':    
            model.classifier[1].weight.requires_grad = True
            model.classifier[1].bias.requires_grad = True

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        logger.critical('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if rank is not None:
                #torch.cuda.set_device(args.gpu)
                model = model.cuda(rank)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = DistributedDataParallel(model, device_ids=[rank])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)

    if torch.cuda.is_available():
        if rank:
            device = torch.device('cuda:{}'.format(rank))
        else:
            device = torch.device("cuda")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    

    main_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    
    warmup_lr_scheduler = LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
    
    scheduler = SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_scheduler], milestones=[args.lr_warmup_epochs])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(rank)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if rank is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = torch.tensor(best_acc1)
                best_acc1 = best_acc1.to(rank)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.critical(f"=> loaded checkpoint '{args.resume}' (epoch {args.start_epoch})")
        else:
            logger.critical(f"=> no checkpoint found at '{args.resume}'")

    # Original directory
    # Train
    original_dataset_path = dataset_root_paths[args.original_dataset]
    original_train_config = dataset_configs[args.original_dataset]['train'][args.original_config]
    original_train_path = os.path.join(original_dataset_path, original_train_config)
    df_original_train = pd.read_pickle(original_train_path)

    # Val needs to be the val_test.pkl set if training MT    
    if args.trainer_type != "DO":    
        original_val_config = dataset_configs[args.original_dataset]['val'][args.original_config]
        original_val_config = original_val_config.replace(".pkl", "_test.pkl")
        original_val_dataset_path = os.path.join(mt_root_directory, "Datasets")
        original_val_path = os.path.join(original_val_dataset_path, original_val_config)
        df_original_val = pd.read_pickle(original_val_path)
    else:
        original_dataset_path = dataset_root_paths[args.original_dataset]
        original_val_config = dataset_configs[args.original_dataset]['val'][args.original_config]
        original_val_path = os.path.join(original_dataset_path, original_val_config)
        df_original_val = pd.read_pickle(original_val_path)

    # Augmented directory
    if args.external_augmentation:        
        df_augmented_train = pd.read_pickle(augmented_df_path)
        frames = [df_original_train, df_augmented_train]    
        df_train = pd.concat(frames)
    else:
        df_train = df_original_train
    
    df_val = df_original_val
    random_seed=args.seed
    # Shuffle the datasets
    df_train = df_train.sample(frac=1, random_state=random_seed)
    df_val = df_val.sample(frac=1, random_state=random_seed)
    df_test = df_val

    # GO-GO-GO!
    # These are imagenet normalizations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    if args.arch == 'resnet50':

        train_dataset = CustomImageDataset(df_train, transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(176),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.RandomErasing(args.random_erasing),
            normalize,
            ]))
        
        val_dataset = CustomImageDataset(df_val, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        test_dataset = CustomImageDataset(df_test, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
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

    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None


    
    num_classes = args.num_classes
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_categories=num_classes, use_v2=args.use_v2
    )
    """
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate
    """
    collate_fn = default_collate

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    

    for epoch in range(args.start_epoch, args.epochs):
        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)
        if(((epoch + 1) % args.num_eval_epochs) == 0):
        # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)
            scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()}, epoch, is_best)

    

    """
    # Test after training
    # Loading the best checkppint
    if not args.eval_pretrained:
        best_ckpt_name = "model_best.pth.tar"
        best_ckpt_path = os.path.join(ckpt_dir, best_ckpt_name)
        logger.critical(best_ckpt_path)
        logger.critical("Testing after training")
        if args.gpu is None:
            checkpoint = torch.load(best_ckpt_path, map_location=loc)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            # Load best checkpoint
            checkpoint = torch.load(best_ckpt_path, map_location=loc)
        best_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        best_acc1 = torch.tensor(best_acc1)
        if args.gpu is not None:
        # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.critical(f"=> loaded checkpoint '{best_ckpt_path}' (epoch {best_epoch})")
    
    validate(test_loader, model, criterion, args)

    ###################################################################################################
    # Code for per-class evaluation 
    logger.critical("Starting per class analysis...")
    if args.test_per_class:        
        
        class_stats = {}
        class_predictions = {}
        class_masks = {}

        unique_classes = df_val['label'].unique()

        for uni_class in unique_classes:
            df_class_val = df_val[df_val['label'] == uni_class]
            # GO-GO-GO!
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

            if args.arch == 'resnet50':
                val_dataset = CustomImageDataset(df_class_val, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
                
            elif args.arch == 'efficientnet_b3':
                val_dataset = CustomImageDataset(df_class_val, transform=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(320, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(300),
                        transforms.ToTensor(),
                        normalize,
                    ]))            
            val_sampler = None
            val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)

            top_1_acc, label_preds, prediction_mask = validate_per_class(val_loader, model, criterion, args)
            class_stats[uni_class] = top_1_acc
            class_predictions[uni_class] = label_preds
            class_masks[uni_class] = prediction_mask

        agg_class_stats = {}
        agg_class_stats['accuracy'] = class_stats
        agg_class_stats['predictions'] = class_predictions
        agg_class_stats['prediction_masks'] = class_masks       
        # Save class_stats
        agg_class_stats_out_path = os.path.join(metrics_dir, "agg_class_stats.pkl")
        with open(agg_class_stats_out_path, 'wb') as o_file:
            pickle.dump(agg_class_stats, o_file)
    ###################################################################################################
    """
    cleanup()

# EDIT HERE

def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 2))
        #f1_score = compute_f1_score(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            stats = progress.display(i + 1)
            logger.critical(stats)
            #logger.critical(f"F1 score is {f1_score}")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def compute_f1_score(preds, targets):
    pred_classes = torch.argmax(preds, dim=1)
    targets = targets.detach().cpu().numpy()
    pred_classes = pred_classes.detach().cpu().numpy()
    f1_res = f1_score(targets, pred_classes, average='micro')
    return f1_res

def validate_per_class(val_loader, model, criterion, args):
    per_instance_preds = Counter()
    prediction_mask = []
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 2))
                pred_classes = torch.argmax(output, dim=1)
                pred_classes = pred_classes.detach().cpu().numpy().tolist()
                labels = target.detach().cpu().numpy().tolist()
                per_instance_preds.update(pred_classes)
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
                
                # Boolean predictions
                pred_mask = np.equal(pred_classes, labels)
                prediction_mask.append(pred_mask)
                
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    stats = progress.display(i + 1)
                    logger.info(stats)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))
    summ_stats = progress.display_summary()
    logger.info(summ_stats)
    return top1.avg, per_instance_preds, prediction_mask[0]

def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 2))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    stats = progress.display(i + 1)
                    logger.critical(stats)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))
    summ_stats = progress.display_summary()
    logger.critical(summ_stats)
    return top1.avg

def save_checkpoint(args, state, epoch, is_best, filename='checkpoint.pth.tar'):
    ckpt_name = f"{epoch+1}_{filename}"
    filename = os.path.join(ckpt_dir, ckpt_name)
    torch.save(state, filename)
    if is_best:
        logger.critical("Saving best model")
        best_ckpt_name = "model_best.pth.tar"
        best_ckpt_path = os.path.join(ckpt_dir, best_ckpt_name)
        shutil.copyfile(filename, best_ckpt_path)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        return ('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))
        return ' '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
    print("Model trained")