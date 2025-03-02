import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import PIL.Image


class enola_dataset(Dataset):
    def __init__(self, args, mode, transform=None):
        self.mode = mode
        if self.mode == 'DO':
            df_path = args.do_train_path
        elif self.mode == 'DHard':
            df_path = args.dhard_path
        self.df = pd.read_pickle(df_path)
        self.transform = transform
        self.classes = sorted(self.df['label'].unique())
    def __len__(self):
        return len(self.df)

    def nb_classes(self):
        return len(self.classes)
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        im = PIL.Image.open(img_path)
        if len(list(im.split())) == 1 : im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        label = self.df.iloc[idx]['label']
        return im, label

    def get_label(self, idx):
        return self.df.iloc[idx]['label']
    def set_subset(self, I):
        pass


"""
class enola_dataset(Dataset):
    def __init__(self, args,mode, transform=None):
        self.mode = mode
        if self.mode == 'DO':
            df_path = args.do_train_path
        elif self.mode == 'DHard':
            df_path = args.dhard_path
        self.df = pd.read_pickle(df_path)
        self.transform = transform
        self.classes = sorted(self.df['label'].unique())

    def __len__(self):
        return len(self.df)

    def nb_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if image.shape[0] == 4:
            image = image[:3, :, :]
        label = self.df.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label
    

"""
