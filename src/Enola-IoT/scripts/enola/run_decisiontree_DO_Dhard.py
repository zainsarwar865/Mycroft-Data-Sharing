import pathlib
import numpy as np
import pandas as pd
import random
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import argparse

from sklearn import tree
from sklearn.tree import export_graphviz
from datetime import date
import os
import pickle
from sklearn import tree
import glob
import argparse





def generate_save_folder(save_path, MT_fn, DO_fn):
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
    return save_path

def convert_to_path(arr):
    path =   ""
    for i in range(len(arr)):
        if arr[i]==1:
            path = path+str(i)+"-"
    return path[:-1]

    
    

def run_decision_tree(DO_fn, Dhard_fn, save_folder, columns):
    last_name = DO_fn.split("/")[-1]
    DO_benign_fn = DO_fn.replace(last_name, "Benign/"+last_name)
    mal_df = pd.read_csv(DO_fn)
    mal_df=mal_df[columns]
    mal_df["Label"]=[1 for i in range(mal_df.shape[0])]
    benign_df = pd.read_csv(DO_benign_fn)
    benign_df=benign_df[columns]
    benign_df["Label"]=[0 for i in range(benign_df.shape[0])]
    df_train = pd.concat([benign_df, mal_df])
    X_train = df_train.drop(columns=["Label"])
    y_train = df_train["Label"]


    clf = DecisionTreeClassifier(random_state=0, max_depth=4)
    clf.fit(X_train, y_train)
    decision_paths_DO = clf.decision_path(mal_df[columns])
    decision_paths_DO_dense = decision_paths_DO.toarray()
    decisionpaths_DO_list = [convert_to_path(arr) for arr in decision_paths_DO_dense]
    mal_df["Decision path"]=decisionpaths_DO_list

    Dhard_df = pd.read_csv(Dhard_fn)
    Dhard_df = Dhard_df[columns]
    decisionpaths_Dhard =  clf.decision_path(Dhard_df)
    decision_paths_Dhard_dense = decisionpaths_Dhard.toarray()
    decisionpaths_Dhard_list = [convert_to_path(arr) for arr in decision_paths_Dhard_dense]
    Dhard_df["Decision path"] = decisionpaths_Dhard_list

    matching_path = []
    for path in decisionpaths_DO_list:
        if path in decisionpaths_Dhard_list:
            matching_path.append(True)
        else:
            matching_path.append(False)
    mal_df["Matching path"] = matching_path
        # Save Model to file
    with open(save_folder+'DO_decisiontree.pkl', 'wb') as file:
        pickle.dump(clf, file)

    mal_df.to_csv(save_folder+"DO_decisiontree_decisions.csv")
    Dhard_df.to_csv(save_folder+"Dhard_decisiontree_decisions.csv")



    












    





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
    parser.add_argument('--Dhard_path',type=str)
    parser.add_argument('--DO_path',type=str)
    parser.add_argument('--MT',type=str)
    parser.add_argument('--DO',type=str)
    parser.add_argument('--search_mode',type=str)


    
    ################
    args = parser.parse_args()
    save_path = args.save_path
    Dhard_path = args.Dhard_path
    DO_path = args.DO_path
    MT = args.MT
    DO = args.DO
    search_mode = args.search_mode

    Dhard_fn = Dhard_path+"/"+MT+".csv"
    if search_mode=="pairwise":
        DO_path_name = DO_path+"/"+DO+".csv"
    elif search_mode=="all":
        DO_path_name = DO_path+"/*.csv"
    for DO_fn in glob.glob(DO_path_name):
        save_folder = generate_save_folder(save_path, Dhard_fn, DO_fn)
        run_decision_tree(DO_fn, Dhard_fn, save_folder, columns)

    





if __name__ == '__main__':
    main()

