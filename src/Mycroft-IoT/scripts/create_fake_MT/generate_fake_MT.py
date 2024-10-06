import pathlib
import numpy as np
import pandas as pd
import random
import glob
import os
import argparse
from itertools import chain, combinations



def generate_fake_MT_from_dir(data_dir):
    save_folder = data_dir+"Dhard/"
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    attack_dir = data_dir+"*/train_val_test/*_val.csv"
 
    for fn in glob.glob(attack_dir):
        
        if "Benign" not in fn:
            malware = fn.split("/")[-3].split("-")[-2]
            attack = fn.split("/")[-1]
            if attack=="PartOfAHorizontalPortScan_val.csv":
                malware = malware+"Part"
            elif attack=="CnC_val.csv":
                malware = malware+"CnC"
            elif attack=="CnC-Torii_val.csv":
                malware = malware+"CnCTorii"
            elif attack=="DDoS_val.csv":
                malware = malware+"DDoS"
            df = pd.read_csv(fn)
            print(save_folder+malware+".csv")
            df.to_csv(save_folder+malware+".csv")

            
            

            
    
      
        



def main():
    parser = argparse.ArgumentParser()
    print("It is here")
    parser.add_argument('--data_dir',type=str)
    
    ################
    args = parser.parse_args()
    data_dir = args.data_dir
    # print(data_dir)
    generate_fake_MT_from_dir(data_dir)


if __name__ == '__main__':
    main()
