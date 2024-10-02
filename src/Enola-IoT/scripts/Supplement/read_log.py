import pandas as pd
import numpy as np
import os

""" 
They also give a log file containing the labels of the flows.
This file convert the log of the flows into csv format
TODO: Please specify the log_fn and save_fn
"""

log_fn = "/Users/tranhongvan/Desktop/Github-Projects/Private-Data-Sharing-new/IoT-data/IoT-23-Dataset/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled"
save_fn = "/Users/tranhongvan/Desktop/Github-Projects/Private-Data-Sharing-new/IoT-data/IoT-23-Dataset/CTU-IoT-Malware-Capture-34-1/label.csv"

############################################
file_ = open(log_fn, 'r')
lines = file_.read().splitlines()

### convert the flows into csv format   ###
fields = lines[6]
types = lines[7]
columns_name = fields.split()[1:]
dat = []

for i in range(8, len(lines)):
    line = lines[i].split()
    dat.append(line)
print(dat[:5])
df = pd.DataFrame(dat, columns = columns_name)
df.to_csv(save_fn, encoding='utf-8')