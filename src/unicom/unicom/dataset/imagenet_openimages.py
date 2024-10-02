import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import PIL.Image
class imagenet_openimages(Dataset):
    def __init__(self, args, mode, transform=None):
        self.root = args.df_root
        self.df_config = args.df_retrieve_config
        self.mode = mode
        if self.mode == 'train':
            df_path = os.path.join(self.root, self.df_config)
        elif self.mode == 'eval':
            df_path = args.df_retrieve_path_val
        self.df = pd.read_pickle(df_path)
        self.transform = transform
        self.classes = sorted(self.df['label'].unique())
    def __len__(self):
        return len(self.df)

    def nb_classes(self):
        return len(self.classes)
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        if self.mode == 'train':
            pass
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
