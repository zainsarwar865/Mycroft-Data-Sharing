import pathlib
import numpy as np
import pandas as pd
import random
from sklearn import svm
import glob
import os
import argparse




def split_df(df,ratio, seed):
    """splitting data according to ratio and seed number, this is often used to get the train dataset"""
    rng = np.random.default_rng(seed)
    train_len=int(ratio*len(df))
    train_indices=rng.choice(len(df),train_len,replace=False)
    test_indices=np.setdiff1d(np.arange(len(df)),train_indices)
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    return train_df, test_df


def split_val_test(df,seed):
    """ splitting the remaining data into val and test set, size of val set depends on the size  of the remaining data
    """
    print(df.shape)
    sample_num = 0
    if df.shape[0]>1000:
        sample_num = df.shape[0]//50
    elif df.shape[0]>200:
        sample_num = df.shape[0]//20
    else:
        sample_num = df.shape[0]//3
    print(sample_num)
    rng = np.random.default_rng(seed)
    val_indices=rng.choice(len(df),sample_num,replace=False)
    print(val_indices)
    test_indices=np.setdiff1d(np.arange(len(df)),val_indices)
    val_df = df.iloc[val_indices]
    test_df = df.iloc[test_indices]
    return val_df, test_df


#############   splitting train, val, test dataframe    ######
    
       


""" This file split each benign and malicious file into train, validation (dhard) and test. 
    Modify train_test_ratio and seed number to change the ratio of data used for training.
"""
def split(data_dir, split_ratio, seed):
    print("start splitting")

    for folder in glob.glob(data_dir+"*/"):
        # create folder to store split data
        save_folder = folder+"train_val_test/"
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        # split each filename, excepted the labeled.csv
        for fn in glob.glob(folder+"*.csv"):
            if fn.split("/")[-1] != "labeled.csv":
                save_name = fn.split("/")[-1].split(".")[0]
                df = pd.read_csv(fn)

                train_df, val_test_df = split_df(df,split_ratio, seed)
                train_df.to_csv(save_folder+save_name+"_train.csv")
          
                val_df, test_df = split_val_test(val_test_df,seed)
                
                val_df.to_csv(save_folder+save_name+"_val.csv")
                test_df.to_csv(save_folder+save_name+"_test.csv")
        




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str)
    parser.add_argument('--split_ratio',type=float)
    parser.add_argument('--seed',type=int)
    ################
    args = parser.parse_args()
    data_dir = args.data_dir
    split_ratio = args.split_ratio
    seed = args.seed
    split(data_dir, split_ratio, seed)


  


if __name__ == '__main__':
    main()
    

  

    
    
    

    