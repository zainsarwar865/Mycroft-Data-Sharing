#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd, numpy as np, glob
import sys
from dateutil.parser import parse
import seaborn as sns, matplotlib.pyplot as plt
from nfstream import NFStreamer, NFPlugin
import matplotlib
matplotlib.rcParams.update({'font.size': 16})



## IGNORE: NFDump, old packet processing tool

file_list = glob.glob("../bas_data/nfdump/*.csv")
df_list = []
for fname in file_list:
    if "current" in fname:
        continue
    print(fname)
    df = pd.read_csv(fname)
    df_list.append(df)

df = pd.concat(df_list)



# create nfstream trace
df = NFStreamer(source="/data/network_data_sharing/BAS_data/trace_2021-04-15_11-14-34_3600.pcap", active_timeout=10, idle_timeout=10).to_pandas()
df.to_csv("nfstream_2021-04-15_11-14-34_3600.csv")



df = pd.read_csv("data/nfstream_2021-04-09_12-41-44_86400.csv")


print(df["bidirectional_bytes"].sum() / (1024*1024*1024))
print(df["src2dst_bytes"].sum() / (1024*1024*1024) , df["dst2src_bytes"].sum() / (1024*1024*1024))

print(df.columns)

df["bidirectional_duration"] = df["bidirectional_duration_ms"]/1000
df["bidirectional_first_seen"] = df["bidirectional_first_seen_ms"].apply(lambda x: int(x/1000))
sns.ecdfplot(data=df["bidirectional_duration"])
plt.grid()
plt.figure()
df["bidirectional_mb"] = df["bidirectional_bytes"]/(1024*1024)
sns.ecdfplot(data=df["bidirectional_mb"])
plt.figure()
df_grp = df.groupby(["bidirectional_first_seen"]).agg({"bidirectional_mb": "sum"}).reset_index()
sns.ecdfplot(data=df_grp["bidirectional_mb"])

df["bidirectional_mb"] = df["bidirectional_bytes"] / (1024*1024) 




bin_size = 300
def timestamp_bin(x):
    return bin_size*int(x/(bin_size*1000))
df["ts_bin"] = df["bidirectional_first_seen_ms"].apply(timestamp_bin)



def getProtocolCategory(x):
    if x == 6:
        return "TCP"
    elif x == 17:
        return "UDP"
    else:
        return "OTHER"  

df['protocol_cat'] = df["protocol"].apply(getProtocolCategory)




df_grp = df.groupby(["ts_bin", "protocol_cat"]).agg({"bidirectional_packets": "sum", "bidirectional_bytes": "sum"}).reset_index()
df_grp["bidirectional_packets"] = df_grp["bidirectional_packets"] / (bin_size*1000*1000)
df_grp["bidirectional_bytes"] = df_grp["bidirectional_bytes"]*8 / (bin_size*1024*1024)




min_ts = df_grp["ts_bin"].min()
df_grp["ts_bin"] = df_grp["ts_bin"].apply(lambda x: x - min_ts)




sns.lineplot(data=df_grp, x="ts_bin", y="bidirectional_bytes", hue="protocol_cat")
plt.grid()
plt.xlabel("timestamp")
plt.ylabel("bitrate (Mbps)")
plt.tight_layout()
plt.savefig("protocol.png")




df_small = df[df["protocol_cat"] == "OTHER"]
df_grp_small = df_small.groupby(["ts_bin", "protocol"]).agg({"bidirectional_packets": "sum", "bidirectional_bytes": "sum"}).reset_index()
df_grp_small["bidirectional_packets"] = df_grp_small["bidirectional_packets"] / (1000*1000)
df_grp_small["bidirectional_bytes"] = df_grp_small["bidirectional_bytes"] / (1024*1024)
df_grp_small["ts_bin"] = df_grp_small["ts_bin"].apply(lambda x: x - min_ts)
sns.lineplot(data=df_grp_small, x="ts_bin", y="bidirectional_bytes", hue="protocol")
plt.grid()
plt.xlabel("timestamp")
plt.ylabel("volume (MB)")
plt.tight_layout()
plt.savefig("other.png")



#df_small = df[df["protocol_cat"] == "OTHER"]
def getApplication(x):
    if x["bidirectional_bytes"] > 100:
        return x["application_name"]
    else:
        return "OTHER"
df_grp_small = df.groupby(["application_name"]).agg({"bidirectional_packets": "sum", "bidirectional_bytes": "sum"}).reset_index()
df_grp_small["bidirectional_packets"] = df_grp_small["bidirectional_packets"] / (1000*1000)
df_grp_small["bidirectional_bytes"] = df_grp_small["bidirectional_bytes"] / (1024*1024)
df_grp_small["application_name_modified"] = df_grp_small.apply(getApplication, axis=1)
sns.barplot(data=df_grp_small, x="application_name_modified", y="bidirectional_bytes")
plt.grid()
plt.xlabel("Application")
plt.ylabel("Volume (MB)")
plt.tight_layout()
plt.savefig("application.png")

sns.lineplot(data=df_grp, x="ts_bin", y="tot_pkt", hue="pr")
plt.grid()
plt.xlabel("timestamp (1-min bin)")
plt.ylabel("num packets (in million)")


df_grp_small = df_grp[df_grp["pr"] != "TCP"]
sns.lineplot(data=df_grp_small, x="ts_bin", y="tot_pkt", hue="pr")
plt.grid()
plt.xlabel("timestamp (1-min bin)")
plt.ylabel("num packets (in million)")


def get_ordered_src_dst(x):
    sorted_x = sorted([x["src_ip"], x["dst_ip"], str(x["src_port"]), str(x["dst_port"])])
    return '_'.join(list(map(str, sorted_x)))

df_new = df
df_new["ordered_tuple"] = df_new.apply(get_ordered_src_dst, axis=1)

df_grp = df_new.groupby("ordered_tuple").agg({"bidirectional_packets": "sum", "bidirectional_bytes": "sum"}).reset_index()
df_grp["bidirectional_packets"] = df_grp["bidirectional_packets"] / (1000*1000)
df_grp["bidirectional_bytes"] = df_grp["bidirectional_bytes"] / (1024*1024)
df_udp = df_new[df_new["protocol"] == 17]
df_grp = df_udp.groupby("ordered_tuple").agg({"bidirectional_packets": "sum", "bidirectional_bytes": "sum"}).reset_index()
df_grp["bidirectional_packets"] = df_grp["bidirectional_packets"] / (1000*1000)
df_grp["bidirectional_bytes"] = df_grp["bidirectional_bytes"] / (1024*1024)
df_grp.sort_values(by="bidirectional_bytes", ascending=False).head(10)
df_grp = df_filter.groupby(["ord_ip_port", "pr"]).agg({"tot_byt": "sum"}).reset_index()
df_grp["tot_byt"] = df_grp["tot_byt"] / (1024*1024) 
print(df_grp.sort_values("tot_byt", ascending=False).head(100))




