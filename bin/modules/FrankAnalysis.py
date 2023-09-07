from __future__ import print_function
from __future__ import division
from past.utils import old_div
from builtins import map
from builtins import zip
import os
import pandas as pd
import shutil
import regex as re
import scipy.stats
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import NNanalysis

def networkOutColNames(model):
    if model=='nnalign':
        return ['Binding_core','Offset','Measure','Prediction',
                  'Peptide','Gap_pos','Gap_lgt','Insert_pos','Insert_lgt',
                  'Core+gap','Reliability','AlleleList_ID','MHC','MHC_prob']
    elif model=='NetMHCIIpan':
        return ['Binding_core','Offset','Measure','Prediction',
                'Peptide','Gap_pos','Gap_lgt','Insert_pos','Insert_lgt',
                'Core+gap','P1_Rel','MHC']
    elif model=='NetMHCIIpan-mal':
        return ['Binding_core','Offset','Measure','Prediction',
                'Peptide','Gap_pos','Gap_lgt','Insert_pos','Insert_lgt',
                'Core+gap','P1_Rel','MHC','AlleleList_ID','Filler']
    elif model=='NetMHCIIpan-mal-hack':
        return ['Binding_core','Offset','Measure','Prediction',
                'Peptide','Gap_pos','Gap_lgt','Insert_pos','Insert_lgt',
                'Core+gap','P1_Rel','AlleleList_ID','MHC','Filler']
    elif model=='nnalign-context':
        return ['Binding_core','Offset','Measure','Prediction',
                  'Peptide','Gap_pos','Gap_lgt','Insert_pos','Insert_lgt',
                  'Core+gap','Reliability','AlleleList_ID','MHC','MHC_prob','Context']
    else:
        sys.exit("Unknown model, cannot estimate header. Exiting")

def networkOut2DF(inDir, inFile,model='nnalign',XAL=False,noSource=False):    
    header = networkOutColNames(model)
    splitLen = len(header)
    dirFile = os.path.join(inDir,inFile)
    lineList = []
    with open(dirFile,'r') as fh:
        for line in fh:
            splitLine = line.split()
            if len(splitLine)==splitLen and '#' not in line:
                lineList.append(splitLine)
    df = pd.DataFrame(lineList,columns = header)
    if noSource:
        df['AlleleList_ID'] = list(['__'.join(x.split('__')[1:]) for x in df['AlleleList_ID'].values])
    if XAL:
        return selectXAL(df,XAL = XAL)
    else:
        return df

def kmerPosNegSplit(df, by = 'Measure'):
    for group in df.groupby(by):
        if float(group[0])==1.0:
            epiDF = group[1]
        else:
            negEpiDF = group[1]
    return negEpiDF,epiDF

def getFrank(pred,df,predCol='Prediction'):
    #Get epitope prediction and negative prediction values
    #Count number of negatives with higher prediction than epitopes, i.e. false positives
    #Return false positive rate
    preds = df[predCol].values
    falsePosBools = [negPred>pred for negPred in preds]
    return old_div(float(sum(falsePosBools)),len(falsePosBools))

def getEpiCoords(epiDF,negEpiDF,kmerCol='Peptide',predCol='Prediction',bindingCore='Binding_core',alleleCol='MHC',alidCol='AlleleList_ID'):
    #epiStats = epiDF.loc[:,[kmerCol,predCol,bindingCore,alleleCol,alidCol]].copy(deep=True)
    epiStats = epiDF.copy(deep=True)
    #epiStats = epiStats.reset_index()
    epiStats['Frank'] = [getFrank(pred,negEpiDF,predCol=predCol)*100 for pred in epiStats[predCol].values]
    epiStats['Frank'] = epiStats['Frank'].apply(lambda frank:round(frank,2))    
    epiStats['IDX'] = epiStats.index
    return epiStats

def getEpitopeStats(kmerDF,splitBy='Measure',kmerCol='Peptide',predCol='Prediction',bindingCore='Binding_core'):
    kmerDF[predCol] = kmerDF[predCol].apply(float)
    negEpiDF, epiDF = kmerPosNegSplit(kmerDF, by = splitBy)
    return getEpiCoords(epiDF,negEpiDF,kmerCol='Peptide',predCol=predCol)

def selectXAL(df, XAL='MAL',valCol1='AlleleList_ID',valCol2='MHC'):
    """Function to select Multi/Single-AlleleLigands(MAL/SAL) from dataframe of peptides"""
    if XAL=='MAL':
        return df[df[valCol1]!=df[valCol2]]
    elif XAL=='SAL':
        return df[df[valCol1]==df[valCol2]]
    else:
        raise ValueError('XAL must be either MAL or SAL!')

def procKmerFilename2(filename,splitter='_',start_i=2,end_i=5):
    filename = os.path.splitext(filename)[0]
    nameSplit = filename.split(splitter)
    if 'DRB' in filename:
        DRBname = '_'.join(nameSplit[start_i:start_i+2])
        return tuple([DRBname]+nameSplit[start_i+2:])
    else:
        return tuple(nameSplit[start_i:end_i])

def procKmerFilename(filename,splitter='_',skip=False,start_i=2,end_i=5):
    filename = os.path.splitext(filename)[0]
    nameSplit = filename.split('---')[1]
    nameSplit = nameSplit.split(splitter)
    peptLen = nameSplit[-1]
    if skip:
        nameSplit = nameSplit[skip:]
    #UID = nameSplit[-2]
    if 'DRB' in filename:
        UID = splitter.join(nameSplit[2:-1])
        allele = splitter.join(nameSplit[:2])
    else:
        UID = splitter.join(nameSplit[1:-1])
        allele = nameSplit[0]
    return allele,UID,peptLen

def procKmerFilename_new(filename,splitter='__',skip=False,start_i=2,end_i=5):
    filename = os.path.splitext(filename)[0]
    nameSplit = filename.split('---')[1]
    nameSplit = nameSplit.split(splitter)
    #print(splitter)
    peptLen = nameSplit[-1]
    UID = nameSplit[-2]
    #alid = nameSplit[-3]
    return UID,peptLen#,alid


def addSourceToStats(epiStats,filename,skip=False,splitter='__'):
    #allele,uid,peptLen = procKmerFilename(filename,skip=skip)
    uid,peptLen = procKmerFilename_new(filename,skip=skip,splitter=splitter)
    numEpi = len(epiStats)
    #epiStats['Allele'] = [allele]*numEpi 
    epiStats['UID'] = uid
    epiStats['Peptlen'] = peptLen
    #epiStats['ALID'] = alid    
    #epiStats['Allele'] = [allele]*numEpi     
    epiStats['Peptlen'] = epiStats['Peptlen'].apply(int)    
    return epiStats

def networkOut2DF_wrap(inDir,inFile,model):
    if model=='NetMHCIIpan-tool':
        header= ['Seq','MHC','Peptide','Identity','Pos','Core','Core_Rel','Prediction','Aff','Rank','Measure','BindingLevel']
        return pd.read_csv(os.path.join(inDir,inFile),sep="\s*",engine='python',quotechar='#',skiprows=11,skipfooter=3,header=None,names=header)
    if model=='NetMHCIIpan-3.2':
        header= ['Seq','MHC','Peptide','Identity','Pos','Core','Core_Rel','1-log50k(aff)','Affinity(nM)','%Rank','Measure','BindingLevel']
        return pd.read_csv(os.path.join(inDir,inFile),sep="\s*",engine='python',quotechar='#',skiprows=11,skipfooter=3,header=None,names=header)
    if model=='NetMHCIIpan-4.0BA':
        header = ['Pos','MHC','Peptide','Of','Core','Core_rel','Identity','Score_EL','%Rank_EL','Measure','Score_BA','Affinity(nM)','%Rank_BA','BindLevel']
        return pd.read_csv(os.path.join(inDir,inFile),sep="\s*",engine='python',quotechar='#',skiprows=13,skipfooter=3,header=None,names=header)
    else:
        return NNanalysis.networkOut2DF(inDir,inFile,model=model)

def kmerFile2epiStats(inDir,inFile,model='nnalign',skip=False,splitter='__',predCol='Prediction'):
    kmerPredDF = networkOut2DF_wrap(inDir,inFile,model)
    try:
        epiStats = getEpitopeStats(kmerPredDF,predCol=predCol)
    except UnboundLocalError as e:
        print("No Epitope found in file: {}".format(os.path.join(inDir,inFile)))
        raise(e)
        
    return addSourceToStats(epiStats,inFile,skip=skip,splitter=splitter)

def kmerFile2predMeas(inDir,inFile,model='nnalign'):
    kmerPredDF = networkOut2DF_wrap(inDir,inFile,model)
    return kmerPredDF[['Measure','Prediction']].values
    #meas = kmerPredDF['Measure']
    #preds = kmerPredDF['Prediction']
    #return list(zip(meas,preds)) 

def removeNonLog(kmer_dir):
    for root, dirs, filenames in os.walk(kmer_dir):
        for filename in filenames:
            if not filename.startswith('log_eval'):
                os.remove(os.path.join(root,filename))

def getEmptyLogs(directory,cond='log_eval'):
    emptyFiles = []
    for root,dirs,files in os.walk(directory):
        for filename in files:
            if cond in filename:
                dirFile = os.path.join(root,filename)
                with open(dirFile, 'r') as fh:
                    lines = fh.readlines()
                    nonCommentCount = sum([not line.startswith('#') for line in lines])
                    if nonCommentCount<10:
                        emptyFiles.append(filename)
    return emptyFiles

def kmerDir2epiStatsDF(inDir,emptyFiles=False,cond='log_eval',model='nnalign',splitter='__',skip=False,predCol='Prediction',rank=False):
    if not emptyFiles:
        emptyFiles = getEmptyLogs(inDir,cond=cond)
    #print(emptyFiles)
    logFiles = [filename for filename in os.listdir(inDir) if cond in filename]
    logFiles = [filename for filename in logFiles if not filename in emptyFiles]
    logFiles = [filename for filename in logFiles if not filename.startswith('.')]
    numLogs = len(logFiles)
    epiStatsList = []
    for i,filename in enumerate(logFiles):
        if i%100==0:
            print('Log File number {} out of {} processed'.format(i,numLogs))
        epiStatsList.append(kmerFile2epiStats(inDir,filename,model=model,skip=skip,splitter=splitter,predCol=predCol))
    outDF = pd.concat(epiStatsList)
    if rank:
        outDF['Frank'] = outDF['Frank'].apply(lambda x:100-x)
    return outDF
    

def kmerDir2measPreds(inDir,emptyFiles,cond='log_eval',model='nnalign'):
    measPredList = []
    for filename in os.listdir(inDir):
        if filename in emptyFiles:#Skip over empty files
            continue
        if cond in filename:
            #print filename
            measPredList.append(kmerFile2predMeas(inDir,filename,model=model))
    return measPredList

def getAssayStats(kmer_dir,model='nnalign',skip=False,splitter='__'):
    assayDFlist = []
    emptyFiles = getEmptyLogs(kmer_dir,cond='log_eval')
    for assayDir in os.listdir(kmer_dir):
        if assayDir.startswith('.'):
            continue
        epiStatDF = kmerDir2epiStatsDF(os.path.join(kmer_dir,assayDir),emptyFiles,model=model,skip=skip,splitter=splitter)
        epiStatDF['Assay'] = [assayDir]*len(epiStatDF)
        print(assayDir)
        assayDFlist.append(epiStatDF)
    return pd.concat(assayDFlist)

def frankDFintersection(df_list, sharedCols=['Peptide','UID','Allele','Assay']):
    #Filter out epitopes that did not finish analysis in any of the experiments
    #This should not be needed
    df_meta = pd.concat(df_list)
    df_shared_meta_list = [group[1] for group in df_meta.groupby(sharedCols) if len(group[1])==len(df_list)]
    return pd.concat(df_shared_meta_list)

def dfQualFilter(df_shared,median=False):
    dfFilterList=[]
    for group in df_shared.groupby(['Dat','OutN']):
        if median:
            dfFilterList.append(group[1][group[1]['Prediction']>group[1]['Prediction'].median()])
        else:
            dfFilterList.append(group[1][group[1]['Frank']<0.2])
    return pd.concat(dfFilterList)

def frankDFintersection_counter(df_qual,num_exp,clustGroup=['Dat','OutN'],sharedCols=['Peptide','UID','Allele','Assay']):
    agreedEpitopes = []
    setNumDict = {}
    for group in df_qual.groupby(sharedCols):
        group[1].sort_values(clustGroup)
        vals = '_'.join(list(['-'.join(x[0]) for x in group[1].groupby(clustGroup)]))
        #vals = '-'.join(group[1][clustGroup].values)
        setNumDict[vals] = setNumDict.get(vals,0)+1
        if len(group[1])==num_exp:
            agreedEpitopes.append(group[1])
    return pd.concat(agreedEpitopes),setNumDict

def frankDFintersection_wrapper(df_list,inCol=False,inVals=False,exCol=False,exVals=False):
    df_shared = frankDFintersection(df_list)
    if inCol:
        df_shared = dfFilter(df_shared,inCol,inVals)
    if exCol:
        df_shared = dfFilter(df_shared,exCol,exVals,exclude=True)
    df_qual = dfQualFilter(df_shared)
    df_intersect,setNumDict = frankDFintersection_counter(df_qual,len(df_list))
    return df_intersect, setNumDict

def frankDFshared_filter(df_list,inCol=False,inVals=False,exCol=False,exVals=False):
    df_shared = frankDFintersection(df_list)
    if inCol:
        df_shared = dfFilter(df_shared,inCol,inVals)
    if exCol:
        df_shared = dfFilter(df_shared,exCol,exVals,exclude=True)
    return df_shared
    
    
def dfFilter(df,col,vals,exclude=False):
    if exclude:
        return df[~df[col].isin(vals)]
    else:
        return df[df[col].isin(vals)]

def dfFilter_inEx(df,inCol,inVals,exCol,exVals):
    df = dfFilter(df,inCol,inVals)
    return dfFilter(df,exCol,exVals,exclude=True)
