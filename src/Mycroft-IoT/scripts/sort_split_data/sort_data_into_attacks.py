import pathlib
import numpy as np
import pandas as pd
import random
import glob
import os
import argparse
""" This file sorts the label according to Benign and malicious. For the malicious, it will sort into each type of attacks and save each file separately
"""

def sort_data_into_attacks(data_dir):
    print("This is data")
    for sub_folder in glob.glob(data_dir+"*/"):
        print(sub_folder)
        fname = "labeled.csv"
        if os.path.exists(sub_folder+fname):
            df = pd.read_csv(sub_folder+fname)
            # print(df["label"].unique())
            benign_df = df[df["label"]=="Benign"]
            benign_df.to_csv(sub_folder+"Benign_only.csv")
            mal_df = df[df["label"]=="Malicious"]
            # print(mal_df["detailed-label"].unique())
            for detailed_label in mal_df["detailed-label"].unique():
                
                sub_df = mal_df[mal_df["detailed-label"]==detailed_label]
                detailed_label = detailed_label.replace("&","n")
                sub_df.to_csv(sub_folder+detailed_label+".csv")
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str)
    
    ################
    args = parser.parse_args()
    data_dir = args.data_dir
    sort_data_into_attacks(data_dir)


if __name__ == '__main__':
    main()
    
    



