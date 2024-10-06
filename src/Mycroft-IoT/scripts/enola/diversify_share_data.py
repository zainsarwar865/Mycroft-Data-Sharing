import pathlib
import numpy as np
import pandas as pd
import random
import os
import glob
import argparse

def sample_diverse(df_all, col_name, sample_num):
    df_all["Index"] = [i for i in range(df_all.shape[0])]
    # Get rid of data alr selected by enola
    df = df_all[df_all["Binning dist"]!= min(df_all["Binning dist"])]

    unique_vals = list(set(df[col_name]))
    vals_list = []
    size_lst = []
    for val in unique_vals:
        df_sub = df[df[col_name]==val]
        vals_list.append(val)
        size_lst.append(df_sub.shape[0])
   
    df_count = pd.DataFrame()
    df_count[col_name] = vals_list
    df_count["Count"] = size_lst
   
    ###### get top 5 popular  binning dist  #####
    sorted_df = df_count.sort_values(by='Count', ascending=False)
    top_values = list(sorted_df[col_name])[:5]
    df_list_diverse = []
    for val in top_values:
        df_sub = df[df[col_name]==val]
        df_list_diverse.append(df_sub)
    # print(binning_dist_popular)
    if len(df_list_diverse)>=1:
        df_diverse = pd.concat(df_list_diverse)
    else:
        df_diverse = pd.DataFrame()
    sample_num = min(df_diverse.shape[0], sample_num)
    sampled_df = df_diverse.sample(n=sample_num, random_state=0)

    return sampled_df


def sample_diverse_path(df_all, col_name, sample_num):
    # Get rid of data alr selected by enola
    df = df_all[df_all["Binning dist"]!= min(df_all["Binning dist"])]

    same_path_df = df[df[col_name]==True]
    sample_num = min(same_path_df.shape[0], sample_num)
    sampled_df_samepath = same_path_df.sample(n=sample_num, random_state=0)

    diff_path_df = df[df[col_name]==False]
    sample_num = min(diff_path_df.shape[0], sample_num)
    sampled_df_diffpath = diff_path_df.sample(n=sample_num, random_state=0)



    return sampled_df_samepath, sampled_df_diffpath





def select_diverse_data(save_folder, binning_path,decisiontree_path, diverse_mode, diverse_num):
    DO_binning_df = pd.read_csv(binning_path+"DO_binning_dist.csv")
    DO_decision_df = pd.read_csv(decisiontree_path+"DO_decisiontree_decisions.csv")
    decision_colname = "Matching path"
    DO_binning_df[decision_colname] = DO_decision_df[decision_colname]
    binning_colname = "Binning dist"
    if diverse_mode == "binning":
        df_diverse = sample_diverse(DO_binning_df, binning_colname, diverse_num)
        sample_size = df_diverse.shape[0]
        df_diverse.to_csv(save_folder+"samplesize_"+str(sample_size)+".csv")
    elif diverse_mode == "decisiontree":
        sampled_df_samepath, sampled_df_diffpath = sample_diverse_path(DO_binning_df, decision_colname, diverse_num)
        sample_size_same = sampled_df_samepath.shape[0]
        sampled_df_samepath.to_csv(save_folder+"samplesize_"+str(sample_size_same)+"_samepath.csv")
        sample_size_diff = sampled_df_diffpath.shape[0]
        sampled_df_diffpath.to_csv(save_folder+"samplesize_"+str(sample_size_same)+"_diffpath.csv")
        
        
def generate_save_folder(save_path, MT, DO, binning_mode, diverse_mode):
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
    save_path = save_path+binning_mode+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path+diverse_mode+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    
    return save_path  

def generate_save_folder(save_path, MT, DO, binning_mode = None, diverse_mode = None):
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
    
    return save_path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--binning_path',type=str)
    parser.add_argument('--decisiontree_path',type=str)
    parser.add_argument('--MT', type=str)
    parser.add_argument('--DO', type=str)
    parser.add_argument('--search_mode', type=str)
    parser.add_argument('--binning_mode', type=str)
    parser.add_argument('--diverse_mode', type=str)
    parser.add_argument('--diverse_num', type=int)
    
    ################
    args = parser.parse_args()
    save_path = args.save_path
    binning_path = args.binning_path
    decisiontree_path = args.decisiontree_path
    MT = args.MT
    DO = args.DO
    search_mode = args.search_mode
    binning_mode = args.binning_mode
    diverse_mode = args.diverse_mode
    diverse_num = args.diverse_num
    if diverse_mode!="none":
        if search_mode=="pairwise":
            binning_path = generate_save_folder(binning_path, MT, DO, binning_mode)
            decisiontree_path = generate_save_folder(decisiontree_path, MT, DO)
            save_folder = generate_save_folder(save_path, MT, DO, binning_mode, diverse_mode)
            select_diverse_data(save_folder, binning_path,decisiontree_path, diverse_mode, diverse_num)

        elif search_mode == "all":
            DO_list = [d.split("/")[-2] for d in glob.glob(binning_path+"/MT*/DO*/")]
            DO_list = [d.split("DO_")[-1] for d in DO_list]
            for DO in set(DO_list):
                print(DO)
                binning_path = generate_save_folder(binning_path, MT, DO, binning_mode)
                decisiontree_path = generate_save_folder(decisiontree_path, MT, DO)
                save_folder = generate_save_folder(save_path, MT, DO, binning_mode, diverse_mode)
                select_diverse_data(save_folder, binning_path,decisiontree_path, diverse_mode, diverse_num)



        

   
        # df_diverse = select_diverse_data(data_path,DO_binning_fn, MT, DO, diverse_mode, diverse_num)









    






if __name__ == '__main__':
    main()