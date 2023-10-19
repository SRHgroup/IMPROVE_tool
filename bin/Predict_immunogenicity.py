
#!/usr/bin/env python3

#python3 bin/Predict_immunogenicity.py --file data/calculated_features_test.tsv --model TME_excluded  --outfile "output_prediction_tme_excluded.tsv" 
#python3 bin/Predict_immunogenicity.py --file data/calculated_features_test.tsv --model TME_included  --outfile "output_prediction_tme_included.tsv" 
#python3 bin/Predict_immunogenicity.py --file data/calculated_features_test.tsv --model Simple  --outfile "output_prediction_simple.tsv" 

# Load global modules 
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFE

# append directories 

# arg parse 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", "-i", type=str, required=True, help="Input file path")
parser.add_argument("--outfile", "-o", type=str, required=True, help="Output file name")
parser.add_argument("--model", "-m", type=str, required=True, choices=["TME_included", "TME_excluded", "Simple"], help="Model selection (TME_included, TME_excluded, Simple)")
args = parser.parse_args()


# Warnings for input files
if not args.file:
    warnings.warn("Input file not provided. Please specify the --file or -i argument.")
if not args.outfile:
    warnings.warn("Output file not provided. Please specify the --outfile or -o argument.")
if not args.model:
    warnings.warn("Model not provided. Please specify the --model or -m argument.")




#aign parsed values 
infile = args.file
output_file = args.outfile
model = args.model

valid_models = ["TME_included", "TME_excluded", "Simple"]
if model not in valid_models:
    warnings.warn(f"Invalid model specified. Please choose one of the following: {', '.join(valid_models)}")



homeDir = '.'

pd.set_option("display.max_columns",999)
pd.set_option("display.max_rows",None)
sns.set_context(context='talk',rc={"lines.linewidth": 2})

df = pd.read_csv(os.path.join(homeDir,infile),sep='\t')



# check all columns is included depending on which model is defined 
cols_to_include = ['Patient','HLA_allele','Mut_peptide',
                  'Aro', 'mw','pI','Inst', 'CysRed','RankEL','RankBA','NetMHCExp',
                  'Expression','SelfSim','Prime','PropHydroAro','HydroCore',
                  'PropSmall','PropAro','PropBasic','PropAcidic','DAI','Stability','Foreigness']


if model == "TME_excluded": 
  add_cols = ['CelPrev','PrioScore']
  cols_to_include.extend(add_cols)          

if model == "TME_included": 
  add_cols = ['CelPrev','PrioScore','CYT','HLAexp','MCPmean']
  cols_to_include.extend(add_cols)
print(cols_to_include)

# check all needed columns exists 
missing_cols = [col for col in cols_to_include if col not in df.columns]
if missing_cols:
    warnings.warn(f"The following columns in the input file are not included in cols_to_include: {', '.join(missing_cols)}")



df_to_model = df[cols_to_include]

print(df_to_model.columns)
print(len(df_to_model))
cols_to_drop = ['Patient', 'HLA_allele', 'Mut_peptide']
df_to_model_numeric = df_to_model.drop(cols_to_drop, axis=1).reset_index(drop=True)
print(df_to_model_numeric.columns)

# model directory 
model_out_dir = os.path.join(homeDir,'models',model)

rf_models = [filename for filename in os.listdir(model_out_dir) if filename.startswith("rf")]

rf_total = list()
rf_models_0 = list(filter(lambda k: 'rf0' in k, rf_models))
rf_models_1 = list(filter(lambda k: 'rf1' in k, rf_models))
rf_models_2 = list(filter(lambda k: 'rf2' in k, rf_models))
rf_models_3 = list(filter(lambda k: 'rf3' in k, rf_models))
rf_models_4 = list(filter(lambda k: 'rf4' in k, rf_models))
rf_total = [rf_models_0,rf_models_1,rf_models_2,rf_models_3,rf_models_4]

# for NA valure replace with the mean of the column 

for i in range(df_to_model_numeric.shape[1]):
    column = df_to_model_numeric.iloc[:, i]
    mean_value = column.mean(skipna=True)
    column.fillna(mean_value, inplace=True)

### run Random forrest 

partition = 0
output_data  = pd.DataFrame()
pred_rf = 0
pred_df = pd.DataFrame()
pred_rf_total = pd.DataFrame()
pred_val_rf = 0
for rf in range(5):
    rf_models = rf_total[rf]
    partition +=1
    av_pred_rf = 0
    num = 0
    for model in rf_models: 
        num += 1
        info = df_to_model[["Patient","HLA_allele", "Mut_peptide"]]
        rf_model = pickle.load(open(model_out_dir +"/"+ model,"rb"))
        predictions_rf_new = rf_model.predict_proba(df_to_model_numeric)
        av_pred_rf += predictions_rf_new
    pred_val_rf = av_pred_rf/num
    pred_rf = pd.DataFrame(pred_val_rf[:,1])
    pred_rf.reset_index(drop=True, inplace=True)
    info.reset_index(drop=True, inplace=True)
    output_data = pd.concat([info,pred_rf],axis = 1)
    output_data["model"] = partition
       # print(bladder_results)
    pred_df = pred_df.append(output_data)

pred_df = pred_df.rename(columns={0: 'prediction_rf'})


pred_df['identity'] = pred_df['Patient'] + '_' + pred_df['HLA_allele'] + '_' + pred_df['Mut_peptide']
pred_df_mean = pred_df.groupby('identity').apply(lambda x: x.assign(mean_prediction_rf=x['prediction_rf'].mean())).drop_duplicates(subset='identity', keep='first').drop('prediction_rf', axis=1)


pred_df_mean_merge = df.merge(pred_df_mean, on=['Patient','HLA_allele','Mut_peptide'], how='left')


# print to outfile 
pred_df_mean_merge.to_csv(os.path.join('.','results',output_file), index=None, sep='\t', mode='w')


