import pandas as pd, numpy as np, glob
import sys, os
from nfstream import NFStreamer, NFPlugin
import glob
import pandas as pd
import numpy as np
import os



""" 
This file helps to convert the raw pcap files into csv of the flows in the pcaps.
TODO: specify the pcap_folder containing the pcaps that needs to  be processed and the csv_folder where the extracted flows need to be stored in
"""
pcap_folder = "/Users/tranhongvan/Desktop/Github-Projects/Private-Data-Sharing-new/private_data_sharing_new/IoT-data/IoT-23-Dataset"
csv_folder = "/Users/tranhongvan/Desktop/Github-Projects/Private-Data-Sharing-new/private_data_sharing_new/IoT-data/IoT-CVS/"

for folder_sub in glob.glob(pcap_folder+"/*/"):
    
    name = folder_sub.split("/")[-2]
    if  not os.path.exists(csv_folder+name):
        print(csv_folder+name)
        os.makedirs(csv_folder+name)
        for fn in glob.glob(folder_sub+"*.pcap"):
            new_name = fn.split("/")[-1].replace("pcap","csv")
            active_timeout, idle_timeout = (120, 30)
            df = NFStreamer(source=fn, active_timeout=active_timeout, idle_timeout=idle_timeout).to_pandas()
            df.to_csv(csv_folder+new_name)
            print(csv_folder+new_name)


