import random
import os, sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import scipy
import matplotlib as mpl
import importlib
import itertools
import shutil

sys.path.append(os.path.join('.','modules'))

import partitionTools
import modelTrainEval
import NNanalysis
import procTools
import pathTools

### Run Old PRIME

def predReadWritePRIME(dfSel,predDir,utilsDir_prime,dataSet='BC',outFileTemplate = 'eval__PRIME__{}__{}.txt',resFileTemplate= 'prediction_{}_PRIME.tsv',tmpDir=False,**kwargs_prime):
    if tmpDir:
        ## This is a hack to get around the problem of PRIME not supporting filenames with spaces
        outDir = predDir        
        predDir=tmpDir
        pathTools.clearDirectory(tmpDir)
    else:
        outDir = predDir
    print("Run PRIME")
    print("Size of input: {}".format(len(dfSel)))
    procTools.groupAlleleRunProc(dfSel,predDir,predDir,peptCol='PeptMut',mhcCol='HLA',srcPath=utilsDir_prime,outFileTemplate = outFileTemplate,printCMD=False,**kwargs_prime)
    
    header_prime = ['Peptide','%Rank_bestAllele','Score_bestAllele','%RankBinding_bestAllele','BestAllele','%Rank_A0101','Score_A0101','%RankBinding_A0101']
    dfPrime = procTools.readNetOutDirApply(predDir,header=header_prime,cols=['Predictor','PeptType','HLA'],skiprows=12,skipfooter=0)
    dfPrime = dfPrime.rename(columns = {'%Rank_bestAllele':'%Rank_PRIME','Score_bestAllele':'Score_PRIME','Peptide':'PeptMut'})

    cols= ['PeptMut','%Rank_PRIME','Score_PRIME','HLA']
    dfPrime[cols].to_csv(os.path.join(outDir,'..',resFileTemplate.format(dataSet)),sep='\t',index=False)
    print("Size of PRIME output: {}".format(len(dfPrime)))
    if tmpDir:
        for file in os.listdir(tmpDir):
            shutil.copyfile(os.path.join(tmpDir,file), os.path.join(outDir,file))
    return dfPrime

### Handle Motifs and generate information content weight matrices

def getPseudoDict(path):
    pseudoDict = {}
    with open(path,'r') as fh:
        for line in fh:
            lineSplit = line.strip().split()
            pseudoDict[lineSplit[0]] = lineSplit[1]
    return pseudoDict

def getBGDict(path):
    bgDict = {}
    with open(path,'r') as fh:
        for line in fh:
            lineSplit = line.strip().split()
            bgDict[lineSplit[0]] = lineSplit[1]
    return bgDict

def readFreqMat(path):
    with open(path,'r') as fh:
        i = 0
        lineList = []
        for line in fh:
            line = line.strip()
            if line.startswith('#'):
                continue
            elif line.startswith('Last'):
                continue
            else:
                i+=1
            if i==1:
                aminos = line.split(' ')
            else:
                lineList.append(line.split(' ')[2:])
    return pd.DataFrame(lineList,columns=aminos)

def getFreqLOdfs(datDir,pseudoseq):
    dfFreq = readFreqMat(os.path.join(datDir,'fmat','{}_EL_100_freq.mat'.format(pseudoseq))).astype(float)
    dfLO = readFreqMat(os.path.join(datDir,'lomat','{}_EL_100.txt'.format(pseudoseq))).astype(float)
    return dfFreq,dfLO

def getInfoDF(dfFreq,dfLO):
    return (dfLO/2*dfFreq)

def getInfoSum(dfInfo):
    return dfInfo.sum(axis=1)

def infoSumWrapper(datDir,pseudoSeq):
    dfFreq,dfLO = getFreqLOdfs(datDir,pseudoSeq)
    dfInfo = getInfoDF(dfFreq,dfLO)
    infoSum = getInfoSum(dfInfo)
    return infoSum.values

def genInfoSumDict(datDir,hlas):
    pseudoDict = getPseudoDict(os.path.join(datDir,'MHC_pseudo.dat'))
    infoSumDict = {}
    for hla in hlas:
        hla = "HLA-{}".format(hla)
        infoSumDict[hla] = infoSumWrapper(datDir,pseudoDict[hla])
    return infoSumDict


def infoSumWrapper_prime(datDir,pseudoSeq):
    dfFreq = readFreqMat(os.path.join(datDir,'fmat','{}_EL_100_freq.mat'.format(pseudoSeq))).astype(float)
    dfInfo = getInfoDF_prime(dfFreq)
    infoSum = getInfoSum_prime(dfInfo)
    return infoSum.values

def getInfoDF_prime(dfFreq):
    return dfFreq*(np.log(dfFreq)/np.log(20))

def getInfoSum_prime(dfInfo):
    return 1+(dfInfo.sum(axis=1))

def genInfoSumDict_prime(datDir,hlas):
    pseudoDict = getPseudoDict(os.path.join(datDir,'MHC_pseudo.dat'))
    infoDict = {}
    for hla in hlas:
        pseudoSeq = pseudoDict["HLA-{}".format(hla)]
        infoDict[hla] = infoSumWrapper_prime(datDir,pseudoSeq)
    return infoDict

def infoDict2df(infoDict):
    return pd.DataFrame(infoDict).T.rename(columns= {i:i+1 for i in range(9)}).sort_index()

def dfInfo2dict(dfInfo):
    return dict(zip(dfInfo.index,dfInfo.values))