import pathlib
import numpy as np
import pandas as pd
import random
import os
import glob
import argparse


def find_best_binning_for_feature(lst):
    """ finding the best binnings to represent the list such that the total number of non-empty <= 5
    """
    lst = list(lst)
    num_bins = [200,100,50,20,10,5]
    for num_b in num_bins:
        hist, bins = np.histogram(lst, bins=num_b)
        non_empty_bins = np.count_nonzero(hist)
        if non_empty_bins <= 6:
            return bins

def get_binning_using_binwidth(lst, width):
    num_bins = int((max(lst) - min(lst)) / width)
    num_bins = max(1, num_bins)
    # print(num_bins)

    hist, bins = np.histogram(lst, bins=num_bins)
    return bins

def sort_data_into_bins(dat, bin_edges):
    dat = list(dat)
    return np.digitize(dat, bin_edges)

def get_smallest_binning(dhard_col, base_col):
    smallest_binnings = []
    for base in base_col:
        dist_lst = [base - hard for hard in dhard_col]
        smallest_binnings.append(min([abs(dist) for dist in dist_lst]))
    return smallest_binnings
        
    
def get_binning_using_DO_Dhard(df_base_col, df_dhard_col, binning_mode):
    if binning_mode=="do":
        bin_edges = find_best_binning_for_feature(df_base_col)
    else:
        print("This mode is DhardDO")
        range_DO = max(df_base_col) - min(df_base_col)
        range_dhard = max(df_dhard_col)- min(df_dhard_col)
        if range_dhard==0:
            bin_edges = find_best_binning_for_feature(df_base_col)
        else:
            range_ratio = range_DO/range_dhard
            if range_ratio<=5:
                bin_edges = find_best_binning_for_feature(df_dhard_col)
                binwidth = bin_edges[1]-bin_edges[0]
                bin_edges = get_binning_using_binwidth(df_base_col, binwidth)
            elif range_ratio<=50:
                binwidth = range_dhard
                bin_edges = get_binning_using_binwidth(df_base_col, binwidth)
            else:
                bin_edges = find_best_binning_for_feature(df_base_col)
                
    return bin_edges
      


        
def find_binning_distance(fn_base, fn_dhard,binning_mode, columns, save_folder):
    df_base = pd.read_csv(fn_base)
    df_dhard = pd.read_csv(fn_dhard)
    df_binning_base = pd.DataFrame()
    df_binning_dhard = pd.DataFrame()
    total_binning_dist = [0 for i in range(df_base.shape[0])]
    for col in columns:
        bin_edges = get_binning_using_DO_Dhard(df_base[col], df_dhard[col], binning_mode)

        base_bin = sort_data_into_bins(df_base[col], bin_edges)
        dhard_bin = sort_data_into_bins(df_dhard[col], bin_edges)

        df_binning_base[col] = base_bin
        df_binning_dhard[col] = dhard_bin
        smallest_from_base_to_dhard = get_smallest_binning(dhard_bin, base_bin)
        total_binning_dist= [total_binning_dist[i] + smallest_from_base_to_dhard[i] for i in range(len(smallest_from_base_to_dhard))]
    df_base["Binning dist"] = total_binning_dist

    df_binning_base.to_csv(save_folder+"DO_binning_coordinates.csv")
    df_binning_dhard.to_csv(save_folder+"Dhard_binning_coordinates.csv")
    df_base.to_csv(save_folder+"DO_binning_dist.csv")
    print("success")

def generate_save_folder(save_path, MT_fn, DO_fn, binning_mode):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if save_path[-1]!="/":
        save_path=save_path+"/"
    MT = MT_fn.split("/")[-1].split(".")[0]
    save_path = save_path+"MT_"+MT+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    DO = DO_fn.split("/")[-1].split(".")[0]
    save_path = save_path+"DO_"+DO+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path+binning_mode+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    return save_path
    
    




columns  = [
       'bidirectional_duration_ms', 'bidirectional_packets',
       'bidirectional_bytes', 
       'src2dst_duration_ms', 'src2dst_packets', 'src2dst_bytes',
       'dst2src_duration_ms',
       'dst2src_packets', 'dst2src_bytes',
       ]





def main():
    parser = argparse.ArgumentParser()
    print("It is here")
    parser.add_argument('--save_path',type=str)
    parser.add_argument('--MT_path',type=str)
    parser.add_argument('--DO_path',type=str)
    parser.add_argument('--MT',type=str)
    parser.add_argument('--DO',type=str)
    parser.add_argument('--search_mode',type=str)
    parser.add_argument('--binning_mode',type=str)


    
    ################
    args = parser.parse_args()
    save_path = args.save_path
    MT_path = args.MT_path
    DO_path = args.DO_path
    MT = args.MT
    DO = args.DO
    search_mode = args.search_mode
    binning_mode = args.binning_mode

    MT_fn = MT_path+"/"+MT+".csv"
    if search_mode=="pairwise":
        DO_path_name = DO_path+"/"+DO+".csv"
    elif search_mode=="all":
        DO_path_name = DO_path+"/*.csv"
    for DO_fn in glob.glob(DO_path_name):
        save_folder = generate_save_folder(save_path, MT_fn, DO_fn, binning_mode)
        find_binning_distance(MT_fn, DO_fn,binning_mode, columns, save_folder)

    





if __name__ == '__main__':
    main()
