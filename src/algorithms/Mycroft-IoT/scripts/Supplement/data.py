import pathlib
import numpy as np
import pandas as pd
import random
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import argparse
import time

seed = 0
rng = np.random.default_rng(seed)

columns_to_keep2 = ['protocol',
       
       'bidirectional_duration_ms', 'bidirectional_packets',
       'bidirectional_bytes', 
       'src2dst_duration_ms', 'src2dst_packets', 'src2dst_bytes',
       'dst2src_duration_ms',
       'dst2src_packets', 'dst2src_bytes',
       'duration', "label","detailed-label"]

columns_to_keep1 = ['protocol',
       'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
       'bidirectional_duration_ms', 'bidirectional_packets',
       'bidirectional_bytes', 'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
       'src2dst_duration_ms', 'src2dst_packets', 'src2dst_bytes',
       'dst2src_first_seen_ms', 'dst2src_last_seen_ms', 'dst2src_duration_ms',
       'dst2src_packets', 'dst2src_bytes', 'ts',
       'duration', "label","detailed-label"]
def relabel_df(df, version):
    if version ==1:
        
        df = df[columns_to_keep1]
    else:
        df = df[columns_to_keep2]
        
    
    new_protocol = []
    for i in df["protocol"]:
        if i==17:
            new_protocol.append(0)
        else:
            new_protocol.append(1)
    df["protocol"]=new_protocol

    label = []
    for i in df["label"]:
        if i=="Benign":
            label.append(0)
        else:
            label.append(1)
    df["label"] = label
    return df

def read_files(args):
    data_source = args.data_source
    train_sources = args.train_sources
    test_sources = args.test_sources
    version = args.version
    df_trains = []
    for fn in train_sources:
        fname = data_source+fn
        df = pd.read_csv(fname)
        df = relabel_df(df,version)
        df_trains.append(df)
    df_train = pd.concat(df_trains)
    df_tests = []
    for fn in test_sources:
        fname = data_source+fn
        df = pd.read_csv(fname)
        df = relabel_df(df,version)
        df_tests.append(df)
    df_test = pd.concat(df_tests)
    return df_train, df_test

def train_and_test(df_train,df_test, message, model_name):
 
    print(message)
    
    X_train = df_train.drop(columns=["Label"]).to_numpy()
    y_train = df_train["Label"].to_numpy()
    X_test = df_test.drop(columns=["Label"]).to_numpy()
    y_test = df_test["Label"].to_numpy()
    df = pd.DataFrame()
    ##############  CLASSIFIERS  ##############
    if model_name =="RF":
        
        n_estimator = []
        params = {"n_estimators":[50,100,150,200,300]}
        classifiers = []
        for param in params["n_estimators"]:
            clf2 = RandomForestClassifier(n_estimators=param, random_state=0)
            classifiers.append(clf2)
            n_estimator.append(param)
        df["n_estimators"] = n_estimator
    elif model_name == "DecisionTree": 
        
        df["max_depth"]=[3,5,7,10]
        classifiers = []
        params = {"max_depth":[3,5,7,10]}
        for param in params["max_depth"]:
            clf2 = DecisionTreeClassifier(random_state=0, max_depth = param)
            classifiers.append(clf2)
    elif model_name == "XGB":
        max_depth = []
        n_estimator = []
        # Model parameters
        params={"max_depth":[15,30], "n_estimators":[15,30,50,100] }
        classifiers = []
        for n in params["max_depth"]:
            for j in params["n_estimators"]:
                
                max_depth.append(n)
                n_estimator.append(j)

                model = XGBClassifier(max_depth=n, n_estimators=j)
                classifiers.append(model)
        df["max_depth"] = max_depth
        df["n_estimators"] = n_estimator


    true_p =[]
    true_n = []
    false_p = []
    false_n = []
    run_time = []
    for clf in classifiers:
        start_time = time.time()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        tp,tn,fp,fn = get_scores(y_predict, y_test)
        true_p.append(tp)
        true_n.append(tn)
        false_p.append(fp)
        false_n.append(fn)
        end_time = time.time()
        run_time.append(end_time-start_time)
    
    
    false_predictions = []
    for i in range(len(false_n)):
        false_predictions.append(false_p[i] + false_n[i])
    best_index = np.argsort(false_predictions)[0]
        
    model = classifiers[best_index]
    
    
    # print(type(end_time))
  
    df["Train time"] = run_time
    df["True Positive"] = true_p
    df["True Negative"] = true_n
    df["False Positive"]=false_p
    df["False Negative"] = false_n
    
    
    return model,df


        


def get_scores(y_predict, y_true):
    y_predict = list(y_predict)
    y_true = list(y_true)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for i in range(len(y_predict)):
        y_p = y_predict[i]
        y_t = y_true[i]
        if y_p == 1 and y_t ==1:
            tp+=1
        elif y_p ==0 and y_t ==0:
            tn +=1
        elif y_p ==1 and y_t == 0:
            fp +=1
        else:
            fn +=1
    return tp,tn,fp,fn    
  
    
    

        
        


   
        
           