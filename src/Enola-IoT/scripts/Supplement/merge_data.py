import pandas as pd

""" 

This file tries to label the extracted flows using the label flowed given by matching these flows based on the IP and port number and ts(timestamp) of the source and destination
TODO: Please specify the path to the extracted flows and the path to the labeld flows.
"""
def cast_float(x):
    try:
        return float(x)
    except:
        return 0
    
def is_overlapping(x):
    s1 = x['bidirectional_first_seen_ms']/1000
    e1 = x['bidirectional_last_seen_ms']/1000
    s2 = x['ts']
    e2 = x['te']
    if (s1 >= s2) and (s1 <= e2):
        return True
    elif (s2 >= s1) and (s2 <= e1):
        return True
    return False

dir = "/Users/tranhongvan/Desktop/Github-Projects/Private-Data-Sharing-new/private_data_sharing_new/IoT-data/IoT-CVS/CTU-IoT-Malware-Capture-34-1/"
fn_extracted = dir+"2018-12-21-15-50-14-192.168.1.195.csv"
fn_labeled_flow = dir+"label.csv"

df_extracted = pd.read_csv(fn_extracted, index_col=None, header=0)
df_labeled_flow = pd.read_csv(fn_labeled_flow, index_col=None, header=0)


df_extracted_copy = df_extracted.copy(deep = True)
df_extracted_copy['src_ip'] = df_extracted['dst_ip']
df_extracted_copy['dst_ip'] = df_extracted['src_ip']
df_extracted_copy['src_port'] = df_extracted['dst_port']
df_extracted_copy['dst_port'] = df_extracted['src_port']
df_extracted = pd.concat([df_extracted,df_extracted_copy])



df_labeled_flow = df_labeled_flow[["ts", 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'label', 'detailed-label','duration']]




labeled_on = ['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p']
extracted_on = ["src_ip", "src_port", "dst_ip", "dst_port"]
df_merge = pd.merge(df_extracted, df_labeled_flow, left_on = extracted_on, right_on=labeled_on)





    
    
df_merge['ts'] = df_merge['ts'].astype(float)
df_merge['duration'] = df_merge["duration"].apply(cast_float)
df_merge['te'] = df_merge["ts"] + df_merge["duration"]
df_merge = df_merge[df_merge.apply(is_overlapping, axis=1)]




final = df_merge[['src_ip', 'src_mac','src_oui',
       'src_port', 'dst_ip', 'dst_mac', 'dst_oui', 'dst_port','protocol','bidirectional_first_seen_ms',
       'bidirectional_last_seen_ms', 'bidirectional_duration_ms',
       'bidirectional_packets', 'bidirectional_bytes', 'src2dst_first_seen_ms',
       'src2dst_last_seen_ms', 'src2dst_duration_ms', 'src2dst_packets',
       'src2dst_bytes', 'dst2src_first_seen_ms', 'dst2src_last_seen_ms',
       'dst2src_duration_ms', 'dst2src_packets', 'dst2src_bytes', 'ts','label', 'detailed-label','duration']]




final.to_csv(dir+"labeled.csv")






