
#!/usr/bin/env python3

# run script
#python3 bin/feature_calculations.py --file data/test_file_for_feature_calculation.tsv --dataset "testing" --ProgramDir "/Users/annieborch/Documents/programs/" --outfile "data/calculated_features_test.tsv" --TmpDir "/Users/annieborch/Documents/programs/"


# ----------
# Load global modules 
import random
import os
import sys
import importlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('bin/src')

import multimerPatientTools
import partitionTools
import procTools
import physiochemical_properties
import kernelSim

# append directories 

# arg parse 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", "-i", type=str, required=True)
parser.add_argument("--outfile", "-o", type=str, required=True)
parser.add_argument("--dataset", "-d", type=str, required=True)
parser.add_argument("--ProgramDir", "-p", type=str, required=True)
parser.add_argument("--TmpDir", "-t", type=str, required=True)
args = parser.parse_args()


#aign parsed values 
infile = args.file
dataset_name = args.dataset
output_file = args.outfile
ProgramDirectory = args.ProgramDir
TemptDirectory = args.TmpDir

homeDir = '.'

pd.set_option("display.max_columns",999)
pd.set_option("display.max_rows",None)
sns.set_context(context='talk',rc={"lines.linewidth": 2})

df = pd.read_csv(os.path.join(homeDir,infile),sep='\t')
# down sample for testrun 

#correct if Norm peptide is NA
df['PeptNorm'] = df.apply(lambda row: row['Mut_peptide'] if pd.isna(row['Norm_peptide']) else row['Norm_peptide'], axis=1)


df = df.rename(columns={'HLA_allele':'MHC',
                            'Mut_peptide':'PeptMut'})
cols = ['MHC', 'PeptMut','PeptNorm']
dfdup = df[cols]
dfdup = dfdup[dfdup. duplicated(keep=False)]
dfSel = df[cols].drop_duplicates()
dfSel['PeptLen'] = dfSel['PeptMut'].apply(len)
dfSel['HLA'] = dfSel['MHC']
dfSel['Patient'] = 1


importlib.reload(multimerPatientTools)
dfMerge = multimerPatientTools.runFeatureGeneration_NetMHCpan_stab_prime_molecular(dfSel,os.path.join(homeDir,'results','predictions'),dataSet=dataset_name,utilsDir = ProgramDirectory,tmpDir = TemptDirectory,runPred=True,plot=False,clean=True)




# make nice names 
dfMerge = dfMerge.rename(columns={'MHC':'HLA_allele','PeptMut':'Mut_peptide',
  'aro':'Aro', 'inst':'Inst', 'cys_red':'CysRed','%Rank_EL_mut':'RankEL','%Rank_EL_wt':'RankEL_wt','%Rank_BA_mut':'RankBA',
                        'Expression_Level':'Expression','Self_Similarity':'SelfSim','Score_PRIME':'Prime',
                        'helix':'PropHydroAro','MeanHydroph_coreNoAnc':'HydroCore', 'MeanHydroph':'HydroAll',
                        'Prop_Small':'PropSmall','Prop_Aromatic':'PropAro','Prop_Basic':'PropBasic',
                        'Prop_Acidic':'PropAcidic','Agrotopicity':'DAI','%Rank_Stab':'Stability'})


cols_to_include =['HLA_allele', 'Mut_peptide', 'PeptNorm', 'PeptLen', 
       'Core', 'Of', 'Gp', 'Gl', 'Ip',
       'Il',  'RankEL', 'RankBA', 'RankEL_wt', 'Stability',
       'Prime', 'DAI','CoreNonAnchor', 'Loci', 'HydroAll', 'HydroCore',
       'PropSmall', 'PropAro', 'PropBasic', 'PropAcidic', 'SelfSim', 'mw',
       'Aro', 'Inst', 'PropHydroAro', 'CysRed', 'pI']
dfMerge = dfMerge[cols_to_include]

df = df.rename(columns={'MHC':'HLA_allele','PeptMut':'Mut_peptide'})



dfMerge_with_input = df.merge(dfMerge, on=['HLA_allele','Mut_peptide','PeptNorm'], how='left')

# correct NA values when MutPeptide is NA 
dfMerge_with_input['RankEL_wt'] = dfMerge_with_input.apply(lambda row: pd.NA if pd.isna(row['Norm_peptide']) else row['RankEL_wt'], axis=1)
dfMerge_with_input['DAI'] = dfMerge_with_input.apply(lambda row: pd.NA if pd.isna(row['Norm_peptide']) else row['DAI'], axis=1)


print(len(dfMerge_with_input))
print(dfMerge_with_input.columns)



# write output file 

dfMerge_with_input.to_csv(os.path.join('.',output_file),sep='\t',index=False)



