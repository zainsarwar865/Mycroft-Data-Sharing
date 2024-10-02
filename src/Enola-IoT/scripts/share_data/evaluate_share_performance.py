import pathlib
import numpy as np
import pandas as pd
import random
import os
import glob
import argparse
import sys

from Supplement.data import train_and_test

# Now you can use your_function() in script2.py


def random_sample(df, sample_size):
    seed = 0
    rng = np.random.default_rng(seed)
    sample_size = min(df.shape[0], sample_size)
    sample_indices=rng.choice(len(df),sample_size,replace=False)
    sample_df = df.iloc[sample_indices]
    return sample_df

def get_MT_data(dat_path, MT, mode):
    attack = ""
    malware = ""
    if "Part" in MT:
        attack = "PartOfAHorizontalPortScan"
        malware = MT.split("Part")[0]
    elif "CnCTorii" in MT:
        attack = "CnC-Torii"
        malware = MT.split("CnCTorii")[0]
    elif "CnC" in MT:
        attack = "CnC"
        malware = MT.split("CnC")[0]
    elif "DDoS" in MT:
        attack = "DDoS"
        malware = MT.split("DDoS")[0]
   
    


    df_list = []
    data_path = dat_path+"CTU-IoT-Malware-Capture-"+malware+"-1/train_val_test/"
    if mode == "train":
        df_list.append(pd.read_csv(data_path+"Benign_only_train.csv"))
        
    elif mode=="test":
        df_benign = pd.read_csv(data_path+"Benign_only_test.csv")
        df_benign["Label"]=[0 for i in range(df_benign.shape[0])]
        df_list.append(df_benign)
        df_attack = pd.read_csv(data_path+attack+"_test.csv")
        df_attack["Label"]=[1 for i in range(df_attack.shape[0])]
        df_list.append(df_attack)
    df = pd.concat(df_list)
    return df
        
    


        


def generate_save_folder(save_path, MT, DO, binning_mode = None, diverse_mode = None, model_name = None,sharing_mode = None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if save_path[-1]!="/":
        save_path=save_path+"/"
    save_path = save_path+"MT_"+MT+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path+"DO_"+DO+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if binning_mode is not None:
        save_path = save_path+binning_mode+"/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    if diverse_mode is not None:
        save_path = save_path+diverse_mode+"/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    if model_name is not None:
        save_path = save_path+model_name+"/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    if sharing_mode is not None:
        save_path = save_path+"sharing-"+sharing_mode+"/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    
    return save_path


def get_enola_data(df, sharing_mode):
    df["Index"] = [i for i in range(df.shape[0])]
    dist_colname = "Binning dist"
    returndf=pd.DataFrame()
    
    if  sharing_mode == "enola":
        smallest_dist = min(df[dist_colname])
        returndf = df[df[dist_colname]==smallest_dist]
        
    elif sharing_mode == "equ_enola":
        smallest_dist = min(df[dist_colname])
        enola_df = df[df[dist_colname]==smallest_dist]
        enola_size = enola_df.shape[0]
        returndf = random_sample(df, enola_size)
        
    elif "enola-sample" in sharing_mode:
        sample_size = int(sharing_mode.split("_")[-1])
        df = df.sample(frac=1).reset_index(drop=True)
        returndf = df.sort_values(by=dist_colname).head(sample_size)

    elif "random-sample" in sharing_mode:
        sample_size = int(sharing_mode.split("_")[-1])
        returndf = random_sample(df, sample_size)
    elif sharing_mode=="full":
        returndf=df
    elif sharing_mode=="nosharing":
        returndf = returndf
    return returndf

def merge_enola_and_diverse(diverse_data, enola_data, columns):
    loc_keep = []
    for i,row in diverse_data.iterrows():
        if row["Index"] not in enola_data["Index"]:
            loc_keep.append(i)
    keep_diverse_df = diverse_data.iloc[loc_keep]
    keep_diverse_df=keep_diverse_df[columns]
    enola_data=enola_data[columns]
    df_combined = pd.concat([keep_diverse_df, enola_data])
    df_combined["Label"] = [1 for i in range(df_combined.shape[0])]
    return df_combined


def get_share_data(binning_path, diverse_path, MT, DO, sharing_mode, binning_mode, diverse_mode, diverse_num, columns):
    binning_fn = generate_save_folder(binning_path, MT, DO, binning_mode)+"DO_binning_dist.csv"
    binning_df = pd.read_csv(binning_fn)
    enola_data = get_enola_data(binning_df, sharing_mode)
    diverse_data=pd.DataFrame()
    if diverse_mode!="none":
        diverse_fn = generate_save_folder(diverse_path, MT, DO, binning_mode, diverse_mode)+"samplesize_" + str(diverse_num)+".csv"
        diverse_data =pd.read_csv(diverse_fn)

    if sharing_mode == "nosharing" or sharing_mode == "full":
        share_data = enola_data
    else:
        share_data = merge_enola_and_diverse(diverse_data, enola_data, columns)
    return share_data


columns  = [
       'bidirectional_duration_ms', 'bidirectional_packets',
       'bidirectional_bytes', 
       'src2dst_duration_ms', 'src2dst_packets', 'src2dst_bytes',
       'dst2src_duration_ms',
       'dst2src_packets', 'dst2src_bytes',
       ]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--dat_path', type=str)
    parser.add_argument('--binning_path',type=str)
    parser.add_argument('--diverse_path',type=str)
    parser.add_argument('--MT', type=str)
    parser.add_argument('--DO', type=str)
    parser.add_argument('--search_mode', type=str)
    parser.add_argument('--sharing_mode', type=str)
    parser.add_argument('--binning_mode', type=str)
    parser.add_argument('--diverse_mode', type=str)
    parser.add_argument('--diverse_num', type=int)
    parser.add_argument('--model_name', type=str)
    
    ################
    args = parser.parse_args()
    save_path = args.save_path
    data_path = args.dat_path
    binning_path = args.binning_path
    diverse_path = args.diverse_path
    MT = args.MT
    DO = args.DO
    search_mode = args.search_mode
    sharing_mode = args.sharing_mode
    binning_mode = args.binning_mode
    diverse_mode = args.diverse_mode
    diverse_num = args.diverse_num
    model_name = args.model_name
    print("Good")



    MT_train_df = get_MT_data(data_path, MT, "train")
  
    MT_train_df= MT_train_df[columns]
    MT_train_df["Label"] = [0 for i in range(MT_train_df.shape[0])]

    MT_test_df = get_MT_data(data_path, MT, "test")


    test_df = MT_test_df[columns]
    test_df["Label"] = MT_test_df["Label"]
    # print(test_df)
    if search_mode == "pairwise":
        shared_data = get_share_data(binning_path, diverse_path, MT, DO, sharing_mode, binning_mode, diverse_mode, diverse_num, columns)
        shared_data = shared_data[columns]
        shared_data["Label"] = [1 for i in range(shared_data.shape[0])]
        train_df = pd.concat([MT_train_df, shared_data])
        
        message = "This is train and test process"
        model,df = train_and_test(train_df, test_df, message, model_name)
        save_folder = generate_save_folder(save_path, MT, DO, binning_mode, diverse_mode, model_name, sharing_mode)
        df.to_csv(save_folder+"diverse_"+str(diverse_num)+".csv")
        print(save_folder+"diverse_"+str(diverse_num)+".csv")
    elif search_mode == "all":
        DO_list = [d.split("/")[-2] for d in glob.glob(binning_path+"/MT*/DO*/")]
        DO_list = [d.split("DO_")[-1] for d in DO_list]
        for DO in set(DO_list):
            shared_data = get_share_data(binning_path, diverse_path, MT, DO, sharing_mode, binning_mode, diverse_mode, diverse_num, columns)
            shared_data = shared_data[columns]
            shared_data["Label"] = [1 for i in range(shared_data.shape[0])]
            train_df = pd.concat([MT_train_df, shared_data])
            
            message = "This is train and test process"
            model,df = train_and_test(train_df, test_df, message, model_name)
            save_folder = generate_save_folder(save_path, MT, DO, binning_mode, diverse_mode, model_name, sharing_mode)
            df.to_csv(save_folder+"diverse_"+str(diverse_num)+".csv")
            print(save_folder+"diverse_"+str(diverse_num)+".csv")








if __name__ == '__main__':
    main()
    
        
