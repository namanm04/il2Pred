#########################################################################################
# il2Pred is developed for predicting the interleukin peptide using hybrid model which #
# combines ML & Motif search approach. It is developed by Prof G. P. S. Raghava's group.#
# Please cite: https://webs.iiitd.edu.in/raghava/il2Pred/                           #
########################################################################################

import argparse
import warnings
import subprocess
import pkg_resources
import os
import sys
import numpy as np
import pandas as pd
import math
import itertools
from collections import Counter
import pickle
import re
import glob
import time
import uuid
from time import sleep
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import urllib.request
import shutil
import zipfile
import joblib

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Please provide the following arguments') 

## Read Arguments from command
parser.add_argument("-i", "--input", type=str, required=True, help="Input: protein or peptide sequence in FASTA format or single sequence per line in single letter code")
parser.add_argument("-o", "--output",type=str, help="Output: File for saving results by default outfile.csv")
parser.add_argument("-m", "--model",type=int, choices = [1,2], help="Model Type: 1: Composition based model, 2: Hybrid Model, by default 1")
parser.add_argument("-t","--threshold", type=float, help="Threshold: Value between 0 to 1 by default 0.5")
parser.add_argument("-d","--display", type=int, choices = [1,2], help="Display: 1:interleukin Peptide only, 2: All Proteins, by default 1")
args = parser.parse_args()
nf_path = os.path.dirname(__file__)

# Function to check the sequence
def readseq(file):
    with open(file) as f:
        records = f.read()
    records = records.split('>')[1:]
    seqid = []
    seq = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', ''.join(array[1:]).upper())
        seqid.append('>'+name)
        seq.append(sequence)
    if len(seqid) == 0:
        f=open(file,"r")
        data1 = f.readlines()
        for each in data1:
            seq.append(each.replace('\n',''))
        for i in range (1,len(seq)+1):
            seqid.append(">Seq_"+str(i))
    df1 = pd.DataFrame(seqid)
    df2 = pd.DataFrame(seq)
    return df1, df2

# Function to check the length of the input sequences
def lenchk(file):
    dflc = file
    dflc.columns = ['Seq']
    for i in range(len(dflc)):
        if len(dflc['Seq'][i]) > 25:
            print("###########################################################################################################################################################")
            print("Error: Please provide sequence(s) with length less than 25. Input sequence at position "+str(i)+" has length "+str(len(dflc['Seq'][i]))+". Please check the sequences.")
            print("###########################################################################################################################################################")
            sys.exit()
        else:
            continue

# Function to split files
def file_split(file):
    df_2, df_3 = readseq(file)
    df1 = pd.concat([df_2, df_3], axis=1)
    df1.columns = ['ID', 'Seq']
    if not os.path.isdir('fasta'):
        os.mkdir(os.getcwd()+'/fasta')
    path = os.getcwd()+'/fasta'
    for i in range(len(df1)):
        df1.loc[i].to_csv(path+'/'+df1['ID'][i].replace('>','')+'.fasta', index=None, header=False, sep="\n")

# Function to generate the features out of sequences
def dpc_comp(seq, out, q=1):
    std = list('ACDEFGHIKLMNPQRSTVWY')
    dd = []
    lengths = []
    
    for i in range(len(seq)):
        cc = []
        seq_length = len(seq[i])
        lengths.append(seq_length)
        
        for j in std:
            for k in std:
                count = 0
                temp = j + k
                for m3 in range(len(seq[i]) - q):
                    b = seq[i][m3:m3 + q + 1:q]
                    b = b.upper()
                    if b == temp:
                        count += 1
                composition = (count / (len(seq[i]) - q)) * 100
                cc.append(composition)
        dd.append(cc)
    
    df2 = pd.DataFrame(dd)
    head = []
    for s in std:
        for u in std:
            head.append("DPC" + str(q) + "_" + s + u)
    df2.columns = head
    
    # Add the Length column
    df2['Length'] = lengths
    df2.to_csv(out, index=None, header=False)
    
def prediction(inputfile, model, out):
    clf = joblib.load(model)
    data_test = np.loadtxt(inputfile, delimiter=',')
    
    # Load the scaler used during training
    scaler = StandardScaler()
    scaler_path = os.path.dirname(model) + "/scaler.pkl"
    scaler = joblib.load(scaler_path)  # Load the scaler from its saved file
    
    # Scale the input data
    X_test = scaler.transform(data_test)
    
    # Make predictions
    y_p_score = clf.predict_proba(X_test)
    y_p_s = y_p_score.tolist()
    df = pd.DataFrame(y_p_s)
    df_1 = df.iloc[:, -1]
    df_1.to_csv(out, index=None, header=False)

def class_assignment(file1, thr,out):
    df1 = pd.read_csv(file1, header=None)
    df1.columns = ['ML Score']
    cc = []
    for i in range(len(df1)):
        if df1['ML Score'][i] >= float(thr):
            cc.append('il2 inducer')
        else:
            cc.append('il2 non-inducer')
    df1['Prediction'] = cc
    df1 = df1.round(3)
    df1.to_csv(out, index=None)

def MERCI_Processor_p(merci_file,merci_processed,name):
    hh =[]
    jj = []
    kk = []
    qq = []
    filename = merci_file
    df = pd.DataFrame(name)
    zz = list(df[0])
    check = '>'
    with open(filename) as f:
        l = []
        for line in f:
            if not len(line.strip()) == 0 :
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    if hh == []:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['0']))
            kk.append('il2 non-inducer')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x:x.strip(),l))[1:])
        df1.columns = ['Name']
        df1['Name'] = df1['Name'].str.strip('(')
        df1[['Seq','Hits']] = df1.Name.str.split("(",expand=True)
        df2 = df1[['Seq','Hits']]
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True,inplace=True)
        total_hit = int(df2.loc[len(df2)-1]['Seq'].split()[0])
        for j in oo:
            if j in df2.Seq.values:
                jj.append(j)
                qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                kk.append('il2 inducer')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('il2 non-inducer')
    df3 = pd.concat([pd.DataFrame(jj),pd.DataFrame(qq),pd.DataFrame(kk)], axis=1)
    df3.columns = ['Name','Hits','Prediction']
    df3.to_csv(merci_processed,index=None)

def Merci_after_processing_p(merci_processed,final_merci_p):
    df5 = pd.read_csv(merci_processed)
    df5 = df5[['Name','Hits']]
    df5.columns = ['Subject','Hits']
    kk = []
    for i in range(0,len(df5)):
        if df5['Hits'][i] > 0:
            kk.append(0.5)
        else:
            kk.append(0)
    df5["MERCI Score Pos"] = kk
    df5 = df5[['Subject','MERCI Score Pos']]
    df5.to_csv(final_merci_p, index=None)

def MERCI_Processor_n(merci_file,merci_processed,name):
    hh =[]
    jj = []
    kk = []
    qq = []
    filename = merci_file
    df = pd.DataFrame(name)
    zz = list(df[0])
    check = '>'
    with open(filename) as f:
        l = []
        for line in f:
            if not len(line.strip()) == 0 :
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    if hh == []:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['1']))
            kk.append('il2 inducer')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x:x.strip(),l))[1:])
        df1.columns = ['Name']
        df1['Name'] = df1['Name'].str.strip('(')
        df1[['Seq','Hits']] = df1.Name.str.split("(",expand=True)
        df2 = df1[['Seq','Hits']]
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True,inplace=True)
        total_hit = int(df2.loc[len(df2)-1]['Seq'].split()[0])
        for j in oo:
            if j in df2.Seq.values:
                jj.append(j)
                qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                kk.append('il2 non-inducer')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('il2 inducer')
    df3 = pd.concat([pd.DataFrame(jj),pd.DataFrame(qq),pd.DataFrame(kk)], axis=1)
    df3.columns = ['Name','Hits','Prediction']
    df3.to_csv(merci_processed,index=None)

def Merci_after_processing_n(merci_processed,final_merci_n):
    df5 = pd.read_csv(merci_processed)
    df5 = df5[['Name','Hits']]
    df5.columns = ['Subject','Hits']
    kk = []
    for i in range(0,len(df5)):
        if df5['Hits'][i] > 0:
            kk.append(-0.5)
        else:
            kk.append(0)
    df5["MERCI Score Neg"] = kk
    df5 = df5[['Subject','MERCI Score Neg']]
    df5.to_csv(final_merci_n, index=None)


def hybrid(ML_output,name1,merci_output_p, merci_output_n,threshold,final_output):
    df6_2 = pd.read_csv(ML_output,header=None)
    df6_1 = pd.DataFrame(name1)
    df5 = pd.read_csv(merci_output_p, dtype={'Subject': object, 'MERCI Score Pos': np.float64})
    df4 = pd.read_csv(merci_output_n, dtype={'Subject': object, 'MERCI Score Neg': np.float64})
    df6 = pd.concat([df6_1,df6_2],axis=1)
    df6.columns = ['Subject','ML Score']
    df6['Subject'] = df6['Subject'].str.replace('>','')
    df7 = pd.merge(df6,df5, how='outer',on='Subject')
    df8 = pd.merge(df7,df4, how='outer',on='Subject')
    df8.fillna(0, inplace=True)
    df8['Hybrid Score'] = df8[['ML Score', 'MERCI Score Pos', 'MERCI Score Neg']].sum(axis=1)
    df8 = df8.round(3)
    ee = []
    for i in range(0,len(df8)):
        if df8['Hybrid Score'][i] > float(threshold):
            ee.append('il2 inducer')
        else:
            ee.append('il2 non-inducer')
    df8['Prediction'] = ee
    df8.to_csv(final_output, index=None)

print('##############################################################################')
print('# The program il2Pred2 is developed for predicting il2 inducer and non il2 inducer #')
print("# peptides from their primary sequence, developed by Prof G. P. S. Raghava's group. #")
print('# ############################################################################')

# Parameter initialization or assigning variable for command level arguments

Sequence= args.input        # Input variable 
 
# Output file 
result_filename = args.output
         
# Threshold 
Threshold= float(args.threshold)

# Model
Model = int(args.model)

# Display
dplay = int(args.display)

print('Summary of Parameters:')
print('Input File: ',Sequence,'; Model: ',Model,'; Threshold: ', Threshold)
print('Output File: ',result_filename,'; Display: ',dplay)

#------------------ Read input file ---------------------
f=open(Sequence,"r")
len1 = f.read().count('>')
f.close()

with open(Sequence) as f:
        records = f.read()
records = records.split('>')[1:]
seqid = []
seq = []
for fasta in records:
    array = fasta.split('\n')
    name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
    seqid.append(name)
    seq.append(sequence)
if len(seqid) == 0:
    f=open(Sequence,"r")
    data1 = f.readlines()
    for each in data1:
        seq.append(each.replace('\n',''))
    for i in range (1,len(seq)+1):
        seqid.append("Seq_"+str(i))

seqid_1 = list(map(">{}".format, seqid))
print(seqid_1)
CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
f.close()
#======================= Prediction Module start from here =====================
if Model==1:
    dpc_comp(seq, 'seq.dpc')
    os.system("perl -pi -e 's/,$//g' seq.dpc")
    prediction('seq.dpc', 'model/dpc_len_model.pkl','seq.pred')
    class_assignment('seq.pred',Threshold,'seq.out')
    df1 = pd.DataFrame(seqid)
    df2 = pd.DataFrame(seq)
    df3 = pd.read_csv("seq.out")
    df3 = round(df3,3)
    df4 = pd.concat([df1,df2,df3],axis=1)
    df4.columns = ['ID','Sequence','ML Score','Prediction']
    df4.loc[df4['ML Score'] > 1, 'ML Score'] = 1
    df4.loc[df4['ML Score'] < 0, 'ML Score'] = 0
    if dplay == 1:
        df4 = df4.loc[df4.Prediction=="il2 inducer"]
    df4.to_csv(result_filename, index=None)
    os.remove('seq.dpc')
    os.remove('seq.pred')
    os.remove('seq.out')

else:
    merci = nf_path + '/merci/MERCI_motif_locator.pl'
    motifs_p = nf_path + '/motifs/pos_motif.txt'
    print(motifs_p)
    motifs_n = nf_path + '/motifs/neg_motif.txt'
    dpc_comp(seq,'seq.dpc')
    os.system("perl -pi -e 's/,$//g' seq.dpc")
    prediction('seq.dpc', 'model/dpc_len_model.pkl','seq.pred')
    os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_p + " -o merci_p.txt -c BETTS-RUSSELL")
    os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o merci_n.txt -c BETTS-RUSSELL")
    MERCI_Processor_p('merci_p.txt','merci_output_p.csv',seqid)
    Merci_after_processing_p('merci_output_p.csv','merci_hybrid_p.csv')
    MERCI_Processor_n('merci_n.txt','merci_output_n.csv',seqid)
    Merci_after_processing_n('merci_output_n.csv','merci_hybrid_n.csv')
    hybrid('seq.pred',seqid,'merci_hybrid_p.csv','merci_hybrid_n.csv',Threshold,'final_output')
    df44 = pd.read_csv('final_output')
    df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
    df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
    if dplay == 1:
        df44 = df44.loc[df44.Prediction=="il2 inducer"]
    else:
        df44 = df44
    df44 = round(df44,3)
    df44.to_csv(result_filename, index=None)
    os.remove('seq.dpc')
    os.remove('seq.pred')
    os.remove('final_output')
    os.remove('merci_hybrid_p.csv')
    os.remove('merci_hybrid_n.csv')
    os.remove('merci_output_p.csv')
    os.remove('merci_output_n.csv')
    os.remove('merci_p.txt')
    os.remove('merci_n.txt')
    os.remove('Sequence_1')


print('\n======= Thanks for using il2Pred. Your results are stored in file :',result_filename,' =====\n\n')
print('Please cite: il2Pred\n')
