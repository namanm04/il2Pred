##############################################################################
#il2pred is developed for predicting IL2 and non IL2      #
#protein from their primary sequence. It is developed by Prof G. P. S.       #
#Raghava's group. Please cite : il2pred                                  #
# ############################################################################
import argparse  
import warnings
import pickle
import os
import re
import sys
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Please provide following arguments. Please make the suitable changes in the envfile provided in the folder.') 

## Read Arguments from command
parser.add_argument("-i", "--input", type=str, required=True, help="Input: protein or peptide sequence in FASTA format or single sequence per line in single letter code")
parser.add_argument("-o", "--output",type=str, default="outfile.csv", help="Output: File for saving results by default outfile.csv")
parser.add_argument("-t","--threshold", type=float, default=0.38, help="Threshold: Value between 0 to 1 by default 0.38")
parser.add_argument("-j","--job", type=int, default=1,choices = [1, 2, 3],help="Job: 1: il2 v non il2, 2: il2 vs other cytokine, 3: il2 vs random")
parser.add_argument("-m","--model",type=int, default=1, choices = [1, 2], help="Model: 1: ET, 2: Hybrid, by default 1")
parser.add_argument("-d","--display", type=int, choices = [1,2], default=2, help="Display: 1:il2, 2: All peptides, by default 2")
args = parser.parse_args()

nf_path = os.path.dirname(__file__)

def onehot(ltr):
    return [1 if i == ord(ltr) else 0 for i in range(5, 123)]

def onehotvec(s):
    return [onehot(c) for c in list(s.lower())]

def get_sequence_length(X):
    return max(len(x) for x in X)

def encode_sequences(X, model, max_seq_length):
    sequence_encode = []

    for i in range(len(X)):
        x = X[i].lower()
        a = onehotvec(x)
        sequence_encode.append(a)

    # Pad or trim sequences to have consistent length of max_seq_length
    sequence_encode = [np.pad(seq, ((0, max_seq_length - len(seq)), (0, 0)), 'constant') for seq in sequence_encode]

    # Convert sequence_encode to a numpy array and cast to float
    sequence_encode = np.array(sequence_encode).astype(np.float32)

    print(sequence_encode.shape)

    # Make predictions using the loaded model
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    predictions = model.run([output_name], {input_name: sequence_encode})

    return predictions[0]

def len_comp(file, out):
    df1 = pd.DataFrame(file, columns=["Seq"])
   
    # Calculate length for each sequence
    df1['Length'] = df1['Seq'].apply(len)
   
    # Create a new DataFrame with just the Length column
    df_length = df1[['Length']]
   
    # Save the lengths to the output file
    df_length.to_csv(out, index=None, header=True)


def dpc_comp(file,out,q=1):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    df1 = pd.DataFrame(file, columns=["Seq"])
    zz = df1.Seq
    dd = []
    for i in range(0,len(zz)):
        cc = []
        for j in std:
            for k in std:
                count = 0
                temp = j+k
                for m3 in range(0,len(zz[i])-q):
                    b = zz[i][m3:m3+q+1:q]
                    b.upper()
                    if b == temp:
                        count += 1
                    composition = (count/(len(zz[i])-(q)))*100
                cc.append(composition)
        dd.append(cc)
    df3 = pd.DataFrame(dd)
    head = []
    for s in std:
        for u in std:
            head.append("DPC"+str(q)+"_"+s+u)
    df3.columns = head
    df3.to_csv(out, index=None, header=True)

def filter_and_scale_data(dpc_file, len_file, selected_cols_file, scaler_path, output_file):
    # Step 1: Read the DPC and Length files
    dpc_data = pd.read_csv(dpc_file, header=0)  # Assumes header exists in dpc
    len_data = pd.read_csv(len_file, header=0)  # Assumes header exists in length
    
    # Step 2: Concatenate the DPC and Length DataFrames
    combined_data = pd.concat([dpc_data, len_data], axis=1)
    
    # Step 3: Load selected columns
    with open(selected_cols_file, 'r') as f:
        selected_columns = [line.strip() for line in f]
    
    # Step 4: Filter to selected columns
    filtered_data = combined_data[selected_columns]
    
    # Step 5: Scale data with loaded scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    scaled_data = scaler.transform(filtered_data)
    
    # Step 6: Save the scaled data without the header
    pd.DataFrame(scaled_data).to_csv(output_file, index=False, header=False)



def prediction(inputfile1, model, out):
    file_name=inputfile1
    file_name1=out
    # Load the model
    clf = joblib.load(model)
   
    # Load the input data (seq.scaled or similar)
    data_test = np.loadtxt(file_name, delimiter=',')
   
    # Prepare the test data
    X_test = data_test
   
    # Get prediction probabilities
    y_p_score1 = clf.predict_proba(X_test)
   
    # Convert the probabilities to a list and extract the last column (assumed to be the class 1 probability)
    y_p_s1 = y_p_score1.tolist()
    df = pd.DataFrame(y_p_s1)
    df_1 = df.iloc[:, -1]
   
    # Save the predictions to the output file
    df_1.to_csv(file_name1, index=None, header=False)

def class_assignment(file1,thr,out):
    df1 = pd.read_csv(file1, header=None)
    df1.columns = ['ML Score']
    cc = []
    for i in range(0,len(df1)):
        if df1['ML Score'][i]>=float(thr):
            cc.append('il2 inducing')
        else:
            cc.append('il2 non-inducing')
    df1['Prediction'] = cc
    df1 =  df1.round(3)
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
            kk.append('il2 non-inducing')
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
                kk.append('il2 inducing')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('il2 non-inducing')
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
    df5["MERCI Score (+ve)"] = kk
    df5 = df5[['Subject','MERCI Score (+ve)']]
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
            qq.append(np.array(['0']))
            kk.append('il2 non-inducing')
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
                kk.append('il2 non-inducing')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('il2 non-inducing')
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
    df5["MERCI Score (-ve)"] = kk
    df5 = df5[['Subject','MERCI Score (-ve)']]
    df5.to_csv(final_merci_n, index=None)


def hybrid(ML_output,name1, seq, merci_output_p, merci_output_n,threshold,final_output):
    df6_3 = pd.read_csv(ML_output,header=None)
    df6_2 = pd.DataFrame(seq)
    df6_1 = pd.DataFrame(name1)
    df5 = pd.read_csv(merci_output_p, dtype={'Subject': object, 'MERCI Score': np.float64})
    df4 = pd.read_csv(merci_output_n, dtype={'Subject': object, 'MERCI Score': np.float64})
    df6 = pd.concat([df6_1,df6_2, df6_3],axis=1)
    df6.columns = ['Subject','Sequence','ML Score']
    df6['Subject'] = df6['Subject'].str.replace('>','')
    df7 = pd.merge(df6,df5, how='outer',on='Subject')
    df8 = pd.merge(df7,df4, how='outer',on='Subject')
    df8.fillna(0, inplace=True)
    df8['Hybrid Score'] = df8[['ML Score', 'MERCI Score (+ve)', 'MERCI Score (-ve)']].sum(axis=1)
    df8 = df8.round(3)
    ee = []
    for i in range(0,len(df8)):
        if df8['Hybrid Score'][i] > float(threshold):
            ee.append('il2 inducing')
        else:
            ee.append('il2 non-inducing')
    df8['Prediction'] = ee
    df8.to_csv(final_output, index=None)

print('##############################################################################')
print('# The program il2pred is developed for predicting il2 inducing and non IL2 #')
print("# peptides from their primary sequence, developed by Prof G. P. S. Raghava's group. #")
print('# ############################################################################')

# Parameter initialization or assigning variable for command level arguments

Sequence= args.input        # Input variable 
 
# Output file 
result_filename = args.output
         
# Threshold 
Threshold= float(args.threshold)

# Job
job = args.job

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
CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
f.close()

if job==1:
    #======================= Prediction Module start from here =====================
    if Model==1:
        dpc_comp(seq, 'seq.dpc')
        len_comp(seq, 'seq.len')
        os.system("perl -pi -e 's/,$//g' " +  "seq.dpc")
        os.system("perl -pi -e 's/,$//g' " + "seq.len")
        filter_and_scale_data(dpc_file= 'seq.dpc', len_file='seq.len', selected_cols_file='model1/dpc_len_feat_sel.txt', scaler_path='model1/dpc_len_feat_scalar.pkl', output_file= 'seq.scaled')
        prediction('seq.scaled', 'model1/dpc_len_feat_sel_model.pkl','seq.pred')
        class_assignment('seq.pred',Threshold, 'seq.out')
        df1 = pd.DataFrame(seqid)
        df2 = pd.DataFrame(seq)
        df3 = pd.read_csv( "seq.out")
        df3 = round(df3,3)
        df4 = pd.concat([df1,df2,df3],axis=1)
        df4.columns = ['Subject','Sequence','ML Score','Prediction']
        df4.loc[df4['ML Score'] > 1, 'ML Score'] = 1
        df4.loc[df4['ML Score'] < 0, 'ML Score'] = 0 
        df4['PPV'] = (df4['ML Score']*1.2341)-0.1182
        df4.loc[df4['PPV'] > 1, 'PPV'] = 1
        df4.loc[df4['PPV'] < 0, 'PPV'] = 0
        df4 = df4.round({'PPV': 3})
        if dplay == 1:
            df4 = df4.loc[df4.Prediction=="il2 inducing"]
        df4.to_csv(result_filename, index=None)
        os.remove('seq.len')
        os.remove('seq.dpc')
        os.remove('seq.pred')
        os.remove('seq.out')
        os.remove('seq.scaled')
        os.remove('Sequence_1')
    elif Model==2:
        merci = nf_path + 'merci/MERCI_motif_locator.pl'
        motifs_p = nf_path + 'motifs1/pos_motif.txt'
        motifs_n = nf_path + 'motifs1/neg_motif.txt'
        dpc_comp(seq,  'seq.dpc')
        len_comp(seq,  'seq.len')
        os.system("perl -pi -e 's/,$//g' " +   "seq.dpc")
        os.system("perl -pi -e 's/,$//g' " +   "seq.len")
        filter_and_scale_data(dpc_file= 'seq.dpc', len_file= 'seq.len', selected_cols_file= 'model1/dpc_len_feat_sel.txt', scaler_path= 'model1/dpc_len_feat_scalar.pkl', output_file= 'seq.scaled')
        prediction( 'seq.scaled', 'model1/dpc_len_feat_sel_model.pkl', 'seq.pred')
        os.system("perl " + merci + " -p " +  "Sequence_1" +  " -i " + motifs_p + " -o " +  "merci_p.txt")
        os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o " +  "merci_n.txt")
        MERCI_Processor_p( "merci_p.txt",  'merci_output_p.csv',seqid)
        Merci_after_processing_p( 'merci_output_p.csv',  'merci_hybrid_p.csv')
        MERCI_Processor_n( "merci_n.txt" , 'merci_output_n.csv',seqid)
        Merci_after_processing_n( 'merci_output_n.csv', 'merci_hybrid_n.csv')
        hybrid( 'seq.pred',seqid, seq,  'merci_hybrid_p.csv',  'merci_hybrid_n.csv',Threshold,  'final_output')
        df44 = pd.read_csv( 'final_output')
        df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        df44['PPV'] = (df44['Hybrid Score']*1.307)-0.1566
        df44.loc[df44['PPV'] > 1, 'PPV'] = 1
        df44.loc[df44['PPV'] < 0, 'PPV'] = 0    
        df44 = df44.round({'PPV': 3})
        if dplay == 1:
            df44 = df44.loc[df44.Prediction=="il2 inducing"]
        else:
            df44 = df44
        df44 = round(df44,3)
        df44.to_csv(result_filename, index=None)
        os.remove('seq.len')
        os.remove('seq.dpc')
        os.remove('seq.pred')
        os.remove('seq.scaled')
        os.remove('Sequence_1')
        os.remove('merci_hybrid_p.csv')
        os.remove('merci_hybrid_n.csv')
        os.remove('merci_output_p.csv')
        os.remove('merci_output_n.csv')
        os.remove('merci_p.txt')
        os.remove('merci_n.txt')
        os.remove('final_output')
    print('\n======= Thanks for using il2pred. Your results are stored in file :',result_filename,' =====\n\n')
    print('Please cite: il2pred\n')
elif job==2:
    #======================= Prediction Module start from here =====================
    if Model==1:
        dpc_comp(seq,  '/seq.dpc')
        len_comp(seq,  '/seq.len')
        os.system("perl -pi -e 's/,$//g' " +   "/seq.dpc")
        os.system("perl -pi -e 's/,$//g' " +   "/seq.len")
        filter_and_scale_data(dpc_file= '/seq.dpc', len_file= '/seq.len', selected_cols_file= 'model2/dpc_len_feat_sel.txt', scaler_path='model2/dpc_len_feat_scalar.pkl', output_file= '/seq.scaled')
        prediction( '/seq.scaled', 'model2/dpc_len_feat_sel_model.pkl', '/seq.pred')
        class_assignment( '/seq.pred',Threshold, '/seq.out')
        df1 = pd.DataFrame(seqid)
        df2 = pd.DataFrame(seq)
        df3 = pd.read_csv( "/seq.out")
        df3 = round(df3,3)
        df4 = pd.concat([df1,df2,df3],axis=1)
        df4.columns = ['Subject','Sequence','ML Score','Prediction']
        df4.loc[df4['ML Score'] > 1, 'ML Score'] = 1
        df4.loc[df4['ML Score'] < 0, 'ML Score'] = 0 
        df4['PPV'] = (df4['ML Score']*1.2341)-0.1182
        df4.loc[df4['PPV'] > 1, 'PPV'] = 1
        df4.loc[df4['PPV'] < 0, 'PPV'] = 0
        df4 = df4.round({'PPV': 3})
        if dplay == 1:
            df4 = df4.loc[df4.Prediction=="il2 inducing"]
        df4.to_csv(result_filename, index=None)
        os.remove('seq.len')
        os.remove('seq.dpc')
        os.remove('seq.pred')
        os.remove('seq.out')
        os.remove('seq.scaled')
        os.remove('Sequence_1')
    elif Model==2:
        merci = nf_path + 'merci/MERCI_motif_locator.pl'
        motifs_p = nf_path + 'motifs2/pos_motif.txt'
        motifs_n = nf_path + 'motifs2/neg_motif.txt'
        dpc_comp(seq,  'seq.dpc')
        len_comp(seq,  'seq.len')
        os.system("perl -pi -e 's/,$//g' " +   "seq.dpc")
        os.system("perl -pi -e 's/,$//g' " +   "seq.len")
        filter_and_scale_data(dpc_file= 'seq.dpc', len_file= '/seq.len', selected_cols_file= 'model2/dpc_len_feat_sel.txt', scaler_path= 'model2/dpc_len_feat_scalar.pkl', output_file= 'seq.scaled')
        prediction( 'seq.scaled','model2/dpc_len_feat_sel_model.pkl', 'seq.pred')
        os.system("perl " + merci + " -p " +  "Sequence_1" +  " -i " + motifs_p + " -o " +  "merci_p.txt")
        os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o " +  "merci_n.txt")
        MERCI_Processor_p( "merci_p.txt",  'merci_output_p.csv',seqid)
        Merci_after_processing_p( 'merci_output_p.csv',  'merci_hybrid_p.csv')
        MERCI_Processor_n( "merci_n.txt" , 'merci_output_n.csv',seqid)
        Merci_after_processing_n( 'merci_output_n.csv', 'merci_hybrid_n.csv')
        hybrid( 'seq.pred',seqid, seq,  'merci_hybrid_p.csv',  'merci_hybrid_n.csv',Threshold,  'final_output')
        df44 = pd.read_csv( 'final_output')
        df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        df44['PPV'] = (df44['Hybrid Score']*1.307)-0.1566
        df44.loc[df44['PPV'] > 1, 'PPV'] = 1
        df44.loc[df44['PPV'] < 0, 'PPV'] = 0    
        df44 = df44.round({'PPV': 3})
        if dplay == 1:
            df44 = df44.loc[df44.Prediction=="il2 inducing"]
        else:
            df44 = df44
        df44 = round(df44,3)
        df44.to_csv(result_filename, index=None)
        os.remove('seq.len')
        os.remove('seq.dpc')
        os.remove('seq.pred')
        os.remove('seq.scaled')
        os.remove('Sequence_1')
        os.remove('merci_hybrid_p.csv')
        os.remove('merci_hybrid_n.csv')
        os.remove('merci_output_p.csv')
        os.remove('merci_output_n.csv')
        os.remove('merci_p.txt')
        os.remove('merci_n.txt')
        os.remove('final_output')
    print('\n======= Thanks for using il2pred. Your results are stored in file :',result_filename,' =====\n\n')
    print('Please cite: il2pred\n')

elif job==3:
    #======================= Prediction Module start from here =====================
    if Model==1:
        dpc_comp(seq,  'seq.dpc')
        len_comp(seq,  'seq.len')
        os.system("perl -pi -e 's/,$//g' " +   "seq.dpc")
        os.system("perl -pi -e 's/,$//g' " +   "seq.len")
        filter_and_scale_data(dpc_file= 'seq.dpc', len_file= 'seq.len', selected_cols_file= 'model3/dpc_len_feat_sel.txt', scaler_path= 'model3/dpc_len_feat_scalar.pkl', output_file= 'seq.scaled')
        prediction( 'seq.scaled','model3/dpc_len_feat_sel_model.pkl', 'seq.pred')
        class_assignment( 'seq.pred',Threshold, 'seq.out')
        df1 = pd.DataFrame(seqid)
        df2 = pd.DataFrame(seq)
        df3 = pd.read_csv( "seq.out")
        df3 = round(df3,3)
        df4 = pd.concat([df1,df2,df3],axis=1)
        df4.columns = ['Subject','Sequence','ML Score','Prediction']
        df4.loc[df4['ML Score'] > 1, 'ML Score'] = 1
        df4.loc[df4['ML Score'] < 0, 'ML Score'] = 0 
        df4['PPV'] = (df4['ML Score']*1.2341)-0.1182
        df4.loc[df4['PPV'] > 1, 'PPV'] = 1
        df4.loc[df4['PPV'] < 0, 'PPV'] = 0
        df4 = df4.round({'PPV': 3})
        if dplay == 1:
            df4 = df4.loc[df4.Prediction=="il2 inducing"]
        df4.to_csv(result_filename, index=None)
        os.remove('seq.len')
        os.remove('seq.dpc')
        os.remove('seq.pred')
        os.remove('seq.out')
        os.remove('seq.scaled')
        os.remove('Sequence_1')
    elif Model==2:
        merci = nf_path + 'merci/MERCI_motif_locator.pl'
        motifs_p = nf_path + 'motifs3/pos_motif.txt'
        motifs_n = nf_path + 'motifs3/neg_motif.txt'
        dpc_comp(seq,  'seq.dpc')
        len_comp(seq,  'seq.len')
        os.system("perl -pi -e 's/,$//g' " +   "seq.dpc")
        os.system("perl -pi -e 's/,$//g' " +   "seq.len")
        filter_and_scale_data(dpc_file= 'seq.dpc', len_file= 'seq.len', selected_cols_file= 'model3/dpc_len_feat_sel.txt', scaler_path= 'model3/dpc_len_feat_scalar.pkl', output_file= 'seq.scaled')
        prediction( 'seq.scaled', 'model3/dpc_len_feat_sel_model.pkl', 'seq.pred')
        os.system("perl " + merci + " -p " +  "Sequence_1" +  " -i " + motifs_p + " -o " +  "merci_p.txt")
        os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o " +  "merci_n.txt")
        MERCI_Processor_p( "merci_p.txt",  'merci_output_p.csv',seqid)
        Merci_after_processing_p( 'merci_output_p.csv',  'merci_hybrid_p.csv')
        MERCI_Processor_n( "/merci_n.txt" , 'merci_output_n.csv',seqid)
        Merci_after_processing_n( '/merci_output_n.csv', 'merci_hybrid_n.csv')
        hybrid( 'seq.pred',seqid, seq,  'merci_hybrid_p.csv',  'merci_hybrid_n.csv',Threshold,  'final_output')
        df44 = pd.read_csv( 'final_output')
        df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        df44['PPV'] = (df44['Hybrid Score']*1.307)-0.1566
        df44.loc[df44['PPV'] > 1, 'PPV'] = 1
        df44.loc[df44['PPV'] < 0, 'PPV'] = 0    
        df44 = df44.round({'PPV': 3})
        if dplay == 1:
            df44 = df44.loc[df44.Prediction=="il2 inducing"]
        else:
            df44 = df44
        df44 = round(df44,3)
        df44.to_csv(result_filename, index=None)
        os.remove('seq.len')
        os.remove('seq.dpc')
        os.remove('seq.pred')
        os.remove('seq.scaled')
        os.remove('Sequence_1')
        os.remove('merci_hybrid_p.csv')
        os.remove('merci_hybrid_n.csv')
        os.remove('merci_output_p.csv')
        os.remove('merci_output_n.csv')
        os.remove('merci_p.txt')
        os.remove('merci_n.txt')
        os.remove('final_output')
    print('\n======= Thanks for using il2pred. Your results are stored in file :',result_filename,' =====\n\n')
    print('Please cite: il2pred\n')
    
