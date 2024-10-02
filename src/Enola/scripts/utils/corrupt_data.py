from torchvision.transforms import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
import cv2
from torchvision.io import read_image
from dataset import CustomImageDataset
import os
from tqdm import tqdm


# Load DO's training data
train_path = "/bigstor/zsarwar/Tsinghua/DF/df_tsinghua_train.pkl"
df_train = pd.read_pickle(train_path)

thrashed_classes = ['Welsh springer spaniel','Bedlington terrier','Pomeranian','pug, pug-dog','African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus','Bernese mountain dog','Boston bull, Boston terrier','Leonberg','Samoyed, Samoyede','borzoi, Russian wolfhound']
df_train = df_train[df_train['class'].isin(thrashed_classes)]

# Load Data you do not want to corrupt
gm_path = "/bigstor/zsarwar/Enola_Augmented/MT_Imagenet_dogs_120_44494fed02dee22a95cced5d0322a3ed/GradMatch/df_Tsinghua_dogs_2efadae5c23906a8816f60f70ddb3532_a00f67a1b22f9f8b51daef2ff8f84bd5_a61f65d07a710a7a3198666a1633db6d.pkl"
unicom_path = "/bigstor/zsarwar/Enola_Augmented/MT_Imagenet_dogs_120_44494fed02dee22a95cced5d0322a3ed/Unicom/df_Tsinghua_dogs_5a2e2fa2a97f1488218c68b2abb7f11a_76b3dee07840bfc89a1ef253b11fa5bc_5dd7ed0dd6661cb0003a956684b7d63e.pkl"
df_gradmatch_retrieved = pd.read_pickle(gm_path)
df_unicom_retrieved = pd.read_pickle(unicom_path)
gm_set = set(df_gradmatch_retrieved.index.to_list())
unicom_set = set(df_unicom_retrieved.index.to_list())
untouched_indices = unicom_set.union(gm_set)
df_untouched = df_train[df_train.index.isin(untouched_indices)]

df_train_corrupt = df_train[~df_train.index.isin(untouched_indices)]

train_dataset = CustomImageDataset(df_train_corrupt.copy(), transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply(torch.nn.ModuleList([
        transforms.GaussianBlur(3, sigma=(0.1, 2)),
        transforms.RandomPerspective(p=0.9),
        transforms.RandomVerticalFlip(p=0.9),
        transforms.RandomHorizontalFlip(p=0.9),
        transforms.RandomAutocontrast(p=1),
        transforms.RandomInvert(0.8),
        transforms.ColorJitter(0.9,0.9,0.9)       
        ]
        ), p=1),
        transforms.RandomApply(torch.nn.ModuleList([
        transforms.RandomPerspective(p=0.7),
        transforms.RandomRotation(45),
        transforms.RandomRotation(165),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomInvert(0.3),
        transforms.ColorJitter(0.9,0.9,0.9)       
        ]
        ), p=1),
        transforms.PILToTensor(),        
        transforms.RandomErasing(p=1.0, scale = (0.34,0.34))
    ]))

new_folder = "high-resolution_corrupted"
for i in tqdm(range(len(df_train_corrupt))):
    img_path = df_train_corrupt.iloc[i]['img_path']
    new_img_name = img_path.split("/")[-1].split(".")[0] + "_noisy.jpg"
    all_folders = img_path.split("/")
    base_path = '/'.join(all_folders[0:5])
    class_folder = all_folders[6]
    new_path = os.path.join(base_path, new_folder, class_folder,  new_img_name)    
    df_train_corrupt.loc[(df_train_corrupt['img_path'] == img_path), 'img_path'] = new_path
    # Need to change the path in this frame 
    img = train_dataset[i][0]
    img = torch.permute(img, (1, 2, 0))
    img = img.detach().cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(new_path, img)
    # Change path in the dataframe
# Save frames

frames = [df_train_corrupt, df_untouched]
df_train = pd.concat(frames)

train_corrupted_path = "/bigstor/zsarwar/Tsinghua/DF/df_tsinghua_train_corrupted.pkl"
df_train.to_pickle(train_corrupted_path)




















"""
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
import cv2
from torchvision.io import read_image
from dataset import CustomImageDataset
import os
from tqdm import tqdm

# Load DO's training data
train_path = "/bigstor/common_data/ISIA-Food-500/DF/df_ISIA-Food-500_train.pkl"
df_train = pd.read_pickle(train_path)
thrashed_classes = ['French onion soup','Hot and sour soup','Pho','Takoyaki', 'Churros','Beignets', 'Lobster bisque', 'Sashimi', 'Clam chowder','Onion rings']
df_train = df_train[df_train['class'].isin(thrashed_classes)]

# Load Data you do not want to corrupt
gm_path = "/bigstor/zsarwar/Enola_Augmented/MT_food101_full_101_f739c19e1aeeaea18c60b1bf802b05db/GradMatch/df_ISIAFood101_full_2efadae5c23906a8816f60f70ddb3532_b9137c400d90d1abccef843237fcb858_fc4e79326c45bb5e832b6abb5659732b.pkl"
unicom_path = "/bigstor/zsarwar/Enola_Augmented/MT_food101_full_101_f739c19e1aeeaea18c60b1bf802b05db/Unicom/df_ISIAFood101_full_5a2e2fa2a97f1488218c68b2abb7f11a_767ab1c7959a7fec38b27535665d1c51_fc4e79326c45bb5e832b6abb5659732b.pkl"
df_gradmatch_retrieved = pd.read_pickle(gm_path)
df_unicom_retrieved = pd.read_pickle(unicom_path)
gm_set = set(df_gradmatch_retrieved.index.to_list())
unicom_set = set(df_unicom_retrieved.index.to_list())
untouched_indices = unicom_set.union(gm_set)
df_untouched = df_train[df_train.index.isin(untouched_indices)]
df_train_corrupt = df_train[~df_train.index.isin(untouched_indices)]

train_dataset = CustomImageDataset(df_train_corrupt.copy(), transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply(torch.nn.ModuleList([
        transforms.GaussianBlur(3, sigma=(0.1, 2)),
        transforms.RandomPerspective(p=0.9),
        transforms.RandomVerticalFlip(p=0.9),
        transforms.RandomHorizontalFlip(p=0.9),
        transforms.RandomAutocontrast(p=1),
        transforms.RandomInvert(0.8),
        transforms.ColorJitter(0.9,0.9,0.9)       
        ]
        ), p=1),
        transforms.RandomApply(torch.nn.ModuleList([
        transforms.RandomPerspective(p=0.7),
        transforms.RandomRotation(45),
        transforms.RandomRotation(165),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomInvert(0.9),
        transforms.ColorJitter(0.9,0.9,0.9)       
        ]
        ), p=1),
        transforms.PILToTensor(),        
        transforms.RandomErasing(p=1.0, scale = (0.36,0.36))
    ]))


new_folder = "ISIA_Food500_corrupted"
for i in tqdm(range(len(df_train_corrupt))):
    img_path = df_train_corrupt.iloc[i]['img_path']
    new_img_name = img_path.split("/")[-1].split(".")[0] + "_noisy.jpg"
    all_folders = img_path.split("/")
    base_path = '/'.join(all_folders[0:4])
    class_folder = '/'.join(all_folders[5:7])
    new_path = os.path.join(base_path, new_folder, class_folder,  new_img_name)
    df_train_corrupt.loc[(df_train_corrupt['img_path'] == img_path), 'img_path'] = new_path
    # Need to change the path in this frame 
    img = train_dataset[i][0]
    img = torch.permute(img, (1, 2, 0))
    img = img.detach().cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(new_path, img)
    # Change path in the dataframe

# Save frames

frames = [df_train_corrupt, df_untouched]
df_train = pd.concat(frames)

train_corrupted_path = "/bigstor/common_data/ISIA-Food-500/DF/df_ISIA-Food-500_train_corrupted.pkl"
df_train.to_pickle(train_corrupted_path)
print("Corrupted DF saved")
"""

"""

from torchvision.transforms import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
import cv2
from torchvision.io import read_image
from dataset import CustomImageDataset
import os
from tqdm import tqdm


# Load DO's training data
train_path = "/bigstor/common_data/UPMC-food-101/DF/df_upmc-food-101_train.pkl"
df_train = pd.read_pickle(train_path)

thrashed_classes = ['French onion soup','Hot and sour soup','Pho','Takoyaki', 'Churros','Beignets', 'Lobster bisque', 'Sashimi', 'Clam chowder','Onion rings']
df_train = df_train[df_train['class'].isin(thrashed_classes)]

# Load Data you do not want to corrupt
gm_path = "/bigstor/zsarwar/Enola_Augmented/MT_food101_full_101_f739c19e1aeeaea18c60b1bf802b05db/GradMatch/df_upmcfood101_full_2efadae5c23906a8816f60f70ddb3532_54f6c6bf7a3d1c1bfa6b4204b66144bb_b36e5a37e79e85ddf0540da91cbdc86c.pkl"
unicom_path = "/bigstor/zsarwar/Enola_Augmented/MT_food101_full_101_f739c19e1aeeaea18c60b1bf802b05db/Unicom/df_upmcfood101_full_5a2e2fa2a97f1488218c68b2abb7f11a_ea422cfafdeca897158dd1c47acc6431_b36e5a37e79e85ddf0540da91cbdc86c.pkl"
df_gradmatch_retrieved = pd.read_pickle(gm_path)
df_unicom_retrieved = pd.read_pickle(unicom_path)
gm_set = set(df_gradmatch_retrieved.index.to_list())
unicom_set = set(df_unicom_retrieved.index.to_list())
untouched_indices = unicom_set.union(gm_set)
df_untouched = df_train[df_train.index.isin(untouched_indices)]

df_train_corrupt = df_train[~df_train.index.isin(untouched_indices)]

train_dataset = CustomImageDataset(df_train_corrupt.copy(), transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply(torch.nn.ModuleList([
        transforms.GaussianBlur(3, sigma=(0.1, 2)),
        transforms.RandomPerspective(p=0.9),
        transforms.RandomVerticalFlip(p=0.9),
        transforms.RandomHorizontalFlip(p=0.9),
        transforms.RandomAutocontrast(p=1),
        transforms.RandomInvert(0.8),
        transforms.ColorJitter(0.9,0.9,0.9)       
        ]
        ), p=1),
        transforms.RandomApply(torch.nn.ModuleList([
        transforms.RandomPerspective(p=0.7),
        transforms.RandomRotation(45),
        transforms.RandomRotation(165),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomInvert(0.9),
        transforms.ColorJitter(0.9,0.9,0.9)       
        ]
        ), p=1),
        transforms.PILToTensor(),        
        transforms.RandomErasing(p=1.0, scale = (0.35,0.35))
    ]))

new_folder = 'train_corrupted'
for i in tqdm(range(len(df_train_corrupt))):
    img_path = df_train_corrupt.iloc[i]['img_path']
    new_img_name = img_path.split("/")[-1].split(".")[0] + "_noisy.jpg"
    all_folders = img_path.split("/")
    base_path = '/'.join(all_folders[0:5])
    class_folder = all_folders[6]
    new_path = os.path.join(base_path, new_folder, class_folder,  new_img_name)    
    df_train_corrupt.loc[(df_train_corrupt['img_path'] == img_path), 'img_path'] = new_path
    # Need to change the path in this frame 
    img = train_dataset[i][0]
    img = torch.permute(img, (1, 2, 0))
    img = img.detach().cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(new_path, img)

# Save frames

frames = [df_train_corrupt, df_untouched]
df_train = pd.concat(frames)

train_corrupted_path = "/bigstor/common_data/UPMC-food-101/DF/df_upmc-food-101_train_corrupted.pkl"
df_train.to_pickle(train_corrupted_path)
print("Saved DF corrupted...")
"""