import pathlib
import numpy as np
import pandas as pd
import random
import glob
import os
import argparse
from itertools import chain, combinations





def powerset(input_list):
    # Use itertools.chain to flatten the combinations and return them as a list of lists
    return list(chain.from_iterable(combinations(input_list, r) for r in range(len(input_list) + 1)))

def create_fakeDO_from_list(tuple_item, save_folder):
    """ get the tuple of filenames, create the fake DO out of it and name it after some naming convention
    """
    if len(tuple_item)>0:
        ## sort the names of the tuple according to attack
        Part_list = []
        CnC_list = []
        CnC_Torii_list = []
        DDoS_list = []
        Benign_list = []
        benign_name_list = [] # this is the benign data belong to the same folder as the attack data
        for fn in tuple_item:
            lastname = fn.split("/")[-1]
            if lastname=="PartOfAHorizontalPortScan_train.csv":
                Part_list.append(fn)
                benign_name_list.append(fn.replace(lastname, "Benign_only_train.csv"))
            elif lastname == "CnC_train.csv":
                CnC_list.append(fn)
                benign_name_list.append(fn.replace(lastname, "Benign_only_train.csv"))
            elif lastname == "CnC-Torii_train.csv":
                CnC_Torii_list.append(fn)
                benign_name_list.append(fn.replace(lastname, "Benign_only_train.csv"))
            elif lastname == "DDoS_train.csv":
                DDoS_list.append(fn)
                benign_name_list.append(fn.replace(lastname, "Benign_only_train.csv"))
            elif "Benign" in lastname:
                Benign_list.append(fn)
        df_list_main = []
        save_name = ""
        ####    generate Fake DO and its naming in order of attack and the malware order    ####
        if len(Part_list)>0:
            malware_lst = []
            df_list = []
            for fn in Part_list:
                malware_num = fn.split("/")[-3].split("-")[-2]
                malware_lst.append(int(malware_num))
                df = pd.read_csv(fn)
                df["Source"] = [malware_num+"-Part" for i in range(df.shape[0])]
                df_list.append(df)
            sorted_indices = np.argsort(malware_num)
            malware_lst_new = [malware_lst[index] for index in sorted_indices]
            df_list_new = [df_list[index] for index in sorted_indices]
            newname = "Part_"+"-".join([str(i) for i in malware_lst_new])
            df_merge = pd.concat(df_list_new)
            save_name = save_name+newname
            df_list_main.append(df_merge)

        if len(CnC_list)>0:
            malware_lst = []
            df_list = []
            for fn in CnC_list:
                malware_num = fn.split("/")[-3].split("-")[-2]
                malware_lst.append(int(malware_num))
                df = pd.read_csv(fn)
                df["Source"] = [malware_num+"-CnC" for i in range(df.shape[0])]
                df_list.append(df)
            sorted_indices = np.argsort(malware_num)
            malware_lst_new = [malware_lst[index] for index in sorted_indices]
            df_list_new = [df_list[index] for index in sorted_indices]
            newname = "CnC_"+"-".join([str(i) for i in malware_lst_new])
            df_merge = pd.concat(df_list_new)
            save_name = save_name+newname
            df_list_main.append(df_merge)
            
        if len(CnC_Torii_list)>0:
            malware_lst = []
            df_list = []
            for fn in CnC_Torii_list:
                malware_num = fn.split("/")[-3].split("-")[-2]
                malware_lst.append(int(malware_num))
                df = pd.read_csv(fn)
                df["Source"] = [malware_num+"-CnCTorii" for i in range(df.shape[0])]
                df_list.append(df)
            sorted_indices = np.argsort(malware_lst)
            malware_lst_new = [malware_lst[index] for index in sorted_indices]
            df_list_new = [df_list[index] for index in sorted_indices]
            newname = "CnCTorii_"+"-".join([str(i) for i in malware_lst_new])
            df_merge = pd.concat(df_list_new)
            save_name = save_name+newname
            df_list_main.append(df_merge)

        if len(DDoS_list)>0:
            malware_lst = []
            df_list = []
            for fn in DDoS_list:
                malware_num = fn.split("/")[-3].split("-")[-2]
                malware_lst.append(int(malware_num))
                df = pd.read_csv(fn)
                df["Source"] = [malware_num+"-DDoS" for i in range(df.shape[0])]
                df_list.append(df)
            sorted_indices = np.argsort(malware_lst)
            malware_lst_new = [malware_lst[index] for index in sorted_indices]
            df_list_new = [df_list[index] for index in sorted_indices]
            newname = "DDoS_"+"-".join([str(i) for i in malware_lst_new])
            df_merge = pd.concat(df_list_new)
            save_name = save_name+newname
            df_list_main.append(df_merge)
        
        
        
        if len(df_list_main)>0:
            
            df_final = pd.concat(df_list_main)
            df_final.to_csv(save_folder+save_name+".csv")
            df_benign_list = []
            for benign_fn in set(benign_name_list):
                df = pd.read_csv(benign_fn)
                df_benign_list.append(df)
            df_benign_final = pd.concat(df_benign_list)
            save_folder_benign = save_folder+"Benign/"
            if not os.path.exists(save_folder_benign):
                os.mkdir(save_folder_benign)
            df_benign_final.to_csv(save_folder_benign+save_name+".csv")




          
   

def generate_fake_DO_from_dir(data_dir):


    save_folder = data_dir+"FakeDO/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    #########   get all possible attacks    ####
    attack_dir = data_dir+"*/train_val_test/*train.csv"
    fn_list = []
    for fn in glob.glob(attack_dir):
        
        if "Benign" not in fn:
            fn_list.append(fn) 
            print(fn)
    
      
    ####    create all possible Fake DO from powerset of filenames  ###
  
    fn_powerset = powerset(fn_list)
 
    for tuple_item in fn_powerset:
        print(tuple_item)
        ##### create a new fake DO from each subset in the powerset ####
        
        create_fakeDO_from_list(tuple_item,save_folder)
        



def main():
    parser = argparse.ArgumentParser()
    print("It is here")
    parser.add_argument('--data_dir',type=str)
    
    ################
    args = parser.parse_args()
    data_dir = args.data_dir
    # print(data_dir)
    generate_fake_DO_from_dir(data_dir)


if __name__ == '__main__':
    main()
