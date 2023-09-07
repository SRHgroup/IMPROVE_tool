from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import zip
from builtins import map
from builtins import object
from past.utils import old_div
import os
import sys
import pandas as pd
import shutil
import regex as re
import scipy.stats
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import random
import time
from Bio import pairwise2
from Bio import Entrez
from Bio import SeqIO
Entrez.email = "birey@bioinformatics.dtu.dk"


def networkOutColNames(model):
    if model=='nnalign':
        return ['Binding_core','Offset','Measure','Prediction',
                  'Peptide','Gap_pos','Gap_lgt','Insert_pos','Insert_lgt',
                  'Core+gap','Reliability','AlleleList_ID','MHC','MHC_prob']
    elif model=='nnalign-context':
        return ['Binding_core','Offset','Measure','Prediction',
                  'Peptide','Gap_pos','Gap_lgt','Insert_pos','Insert_lgt',
                  'Core+gap','Reliability','AlleleList_ID','MHC','MHC_prob','Context']
    if model=='nnalign-rank':
        return ['Binding_core','Offset','Measure','Prediction','Prediction_Raw',
                  'Peptide','Gap_pos','Gap_lgt','Insert_pos','Insert_lgt',
                  'Core+gap','Reliability','AlleleList_ID','MHC','MHC_prob']
    elif model=='nnalign-rank-context':
        return ['Binding_core','Offset','Measure','Prediction','Prediction_Raw',
                  'Peptide','Gap_pos','Gap_lgt','Insert_pos','Insert_lgt',
                  'Core+gap','Reliability','AlleleList_ID','MHC','MHC_prob','Context']
    elif model=='mixmhc2pred':
        return ['Peptide','Measure','Allele','BestAllele','%Rank','%Rank_perL','BestCore','Best_s','Prediction','MHC','Offset','Binding_core']
    
    elif model=='nnalign_old':
        return ['Binding_core','Offset','Measure','Prediction',
                  'Peptide','Gap_pos','Gap_lgt','Insert_pos','Insert_lgt',
                  'Core+gap','Reliability','MHC']
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
    elif model=='NetMHCpan':
    	return ['Pos','HLA','Peptide','Core',
    			'Of','Gp','Gl','Ip','Il','Icore',
    			'Identity','Score','BindLevel']
    elif model=='NetMHCpan-4.0':
        return ['Pos','HLA','Peptide','Core',
                'Of','Gp','Gl','Ip','Il','Icore',
                'Identity','Score Aff(nM)','%Rank','Exp','BindLevel']
    elif model=='NetSurfP':
        return ['Class Assignment','Amino Acid','Sequence Name',
                'Amino Acid Number','RSA','ASA','NA',
                'Palpha','Pbeta','Pcoil']
    elif model=='MHCnuggets':
        return ['Peptide','Prediction','Measure','MHC']
    elif model=='DeepSeqPanII':
        return ['Date','IEDB reference','MHC','Peptide length','Measurement type','Peptide','Measure','Prediction','NN-align','NetMHCIIpan-3.1','Comblib matrices','SMM-align','Tepitope (Sturniolo)','Consensus IEDB method']
    else:
        sys.exit("Unknown model: {}, cannot estimate header. Exiting".format(model))
    
def concatSelectFiles(inDir,outDir,outFile,selection,splitLen=14,verbose=False):
    regSelect = re.compile(selection)
    open(os.path.join(outDir,outFile),'w').close()#Empty outfile
    with open(os.path.join(outDir,outFile),'a') as fo:
        for root, dirs, files in os.walk(inDir):
            if root==outDir and inDir!=outDir:
                continue
            for filename in sorted(files):
                if re.match(regSelect, filename) is not None:
                    if verbose:
                        print(os.path.join(root,filename))
                    with open(os.path.join(root,filename),'r') as fi:
                        for line in fi:
                            splitLine = line.split()
                            if len(splitLine)==splitLen and '#' not in line:
                                fo.write(line)

def concatSelectFiles_model(inDir,outDir,outFile,selection,splitLen=14,verbose=False):
    regSelect = re.compile(selection)
    open(os.path.join(outDir,outFile),'w').close()#Empty outfile
    
    splitLen = len(networkOutColNames(model))
    
    with open(os.path.join(outDir,outFile),'a') as fo:
        for root, dirs, files in os.walk(inDir):
            if root==outDir and inDir!=outDir:
                continue
            for filename in sorted(files):
                if re.match(regSelect, filename) is not None:
                    if verbose:
                        print(os.path.join(root,filename))
                    with open(os.path.join(root,filename),'r') as fi:
                        for line in fi:
                            splitLine = line.split()
                            if len(splitLine)==splitLen and '#' not in line:
                                fo.write(line)
                                
def networkOut2DF(inDir, inFile='log_eval_concat.txt',model='nnalign',XAL=False,noSource=False,pos=False,predThresh=False,predCol='Prediction',rank=False):    
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
        df['AlleleList_ID'] = ['__'.join(x.split('__')[1:]) for x in df['AlleleList_ID'].values]
    if XAL:
        df = selectXAL(df,XAL = XAL)
    if pos:
        df = networkOut2pos(df)
    if predThresh:
        df = networkOutPredThresh(df,predThresh=predThresh,predCol=predCol,rank=rank)
    try:
        df[predCol] = df[predCol].apply(float)
        df['Measure'] = df['Measure'].apply(float)
    except KeyError:
        pass
    if 'Prediction_Raw' in df.columns:
        df['Prediction_Raw'] = df['Prediction_Raw'].apply(float)
    return df

    
def cvDir2concatDF(evalDir,outDir=False,selection='log_eval\d+-\d\.txt',model='nnalign',
                   concatFile='log_eval_concat.txt',pos=False):
    header = networkOutColNames(model)
    splitLen = len(header)
    if not outDir:
        outDir=evalDir
    concatSelectFiles(evalDir,outDir,concatFile,selection,splitLen)
    concatFile_path = os.path.join(outDir,concatFile)
    df = pd.read_csv(concatFile_path,names=header,sep=' ',index_col=False)
    if pos:
        df = networkOut2pos(df)
    return df

def cvDir2resultClass(evalDir,outDir=False,selection='log_eval\d+-\d\.txt',
                      model='nnalign',datType='EL',concatFile='log_eval_concat.txt',
                     splitter='MHC',noSource=False,XAL=False,peptNumThresh=False,predCol='Prediction'):
    if not outDir:
        outDir = evalDir
    df_eval = cvDir2concatDF(evalDir,outDir,selection=selection,model=model,concatFile=concatFile)
    if noSource:
        df_eval['AlleleList_ID'] = ['__'.join(x.split('__')[1:]) for x in df_eval['AlleleList_ID'].values]
    if XAL:
        df_eval = selectXAL(df_eval,XAL=XAL)
    return NNalignOutput_splitter(df_eval,splitter,datType=datType,peptNumThresh=peptNumThresh,predCol=predCol)

def cvDir2summaryDF(root,direct,selection='log_eval\d+-\d\.txt',
                      model='nnalign',datType='EL',concatFile='log_eval_concat.txt',
                   splitter='MHC',noSource=False,XAL=False,peptNumThresh=False,predCol='Prediction'):
    evalDir = os.path.join(root,direct)
    resClass = cvDir2resultClass(evalDir,evalDir,selection=selection,
                                   model=model,datType=datType,
                                 splitter=splitter,noSource=noSource,XAL=XAL,
                                 peptNumThresh=peptNumThresh,predCol=predCol)
    
    summaryDF = resClass.summaryDF
    #if len(splitter)==1:
    #    summaryDF = summaryDF.reset_index().rename(columns={"index":splitter[0]})
    #else:
    #    summaryDF = summaryDF.reset_index().rename(columns={"level_{}".format(i):s for i,s in enumerate(splitter)})
    return summaryDF

def networkOut2resultClass(inDir,inFile,model='nnalign',
                           datType='BA',splitter='MHC',XAL=False,predCol='Prediction'):
    df_eval = networkOut2DF(inDir, inFile,model=model)
    if XAL:
        df_eval = selectXAL(df_eval,XAL=XAL)
    return NNalignOutput_splitter(df_eval,splitter,datType=datType,predCol=predCol)

def networkOut2summaryDF(inDir,inFile,model='nnalign',
                           datType='BA',splitter='MHC',XAL=False,
                           writeSplit=False,predCol='Prediction'):
    outClass = networkOut2resultClass(inDir,inFile,model=model,
                                    datType=datType,splitter=splitter,XAL=XAL,predCol=predCol)
    if writeSplit:
        splitDir = os.path.join(inDir,'writeSplit')
        outClass.writeSplit(splitDir,onlyPos=True,col='Binding_core')
    return outClass.summaryDF

def selectXAL(df, XAL='MAL',valCol1='AlleleList_ID',valCol2='MHC'):
    """Function to select Multi/Single-AlleleLigands(MAL/SAL) from dataframe of peptides"""
    #This is very fragily, does not work when source is in ALID
    if XAL=='MAL':
        return df[df[valCol1]!=df[valCol2]]
    elif XAL=='SAL':
        return df[df[valCol1]==df[valCol2]]
    else:
        raise ValueError('XAL must be either MAL or SAL!')

def networkOut2pos(df,meas='Measure',pred='Prediction'):
    return df[list([float(x)==1.0 for x in df[meas].values])]

def networkOutPredThresh(df,predThresh=0.01,predCol='Prediction',rank=False):
    if rank:
        return df[df[predCol].apply(lambda pred: float(pred)<predThresh)]
    else:
        return df[df[predCol].apply(lambda pred: float(pred)>predThresh)]

def getTargetPred(df, targCol='Measure',predCol = 'Prediction'):
    return df[targCol], df[predCol]

def getAUC_EL(y,pred):
    y_int = [int(i) for i in y]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_int, pred)
    auc = sklearn.metrics.auc(fpr, tpr)
    return fpr,tpr,auc

def getAUC_BA(y,pred,bind_thresh = 0.4256251898085073): #500nm IC50 threshold for binding, transformed to range [0,1]
    y_bool = [1 if i>bind_thresh else 0 for i in y]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_bool, pred)
    auc = sklearn.metrics.auc(fpr, tpr)
    return fpr,tpr,auc


def isBAbinder(x,bind_thresh = 0.4256251898085073): #500nm IC50 threshold for binding, transformed to range [0,1]
    return int(x>bind_thresh)

def affTransInv(affinityPred):
    """Transform BA binding value in range 0-1 to range 1-50000nm"""
    return 50000**(1-affinityPred)

def affTrans(affinityNm):
    #Apply normalization function as defined by Morten in some early publication 
    #- find that publication
    if affinityNm < 1:
        return 1.0
    elif affinityNm > 50000:
        return 0.0
    else:
        return 1 - old_div(np.log10(affinityNm),np.log10(50000))

def getPCC(y,pred):
    PCC = scipy.stats.pearsonr(y,pred)[0]
    return round(PCC,3)

#def getAUCThresh(y,pred,thresh=0.1):
#    thresh = int(100*thresh)
#    zipPred = zip(pred,y)
#    zipPred.sort(reverse=True)
#    numPred = len(zipPred)
#    pred01,y01 = zip(*zipPred[:numPred/thresh])
#    return getAUC_EL(y01,pred01)

def getAUCThresh(y,pred,thresh=0.1):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred)
    prTup = list(zip(fpr,tpr))
    idx = 0
    for i,pr in enumerate(prTup):
        if pr[0]>=thresh:
            idx=i
            break
    fpr_thresh, tpr_thresh = list(zip(*prTup[:idx]))
    auc_thresh = old_div(sklearn.metrics.auc(fpr_thresh,tpr_thresh),fpr_thresh[idx-1])
    return fpr_thresh, tpr_thresh, round(auc_thresh,3)

def getPPV_pos(y,pred,posNum):
    zipPred = list(zip(pred,y))
    zipPred.sort(reverse=True)
    numPred = len(zipPred)
    pred_top,y_top = list(zip(*zipPred[:posNum]))
    top_pos = sum([int(i) for i in y_top])
    PPV = float(top_pos)/posNum
    return round(PPV,3)

def plotAUCsummary(plotAUCdat,outDirect, outFile):
    samples = [dat[0] for dat in plotAUCdat]
    #print samples
    sample2col, color_list,patches = sample2paperCol(samples)
    
    for i,dat in enumerate(plotAUCdat):
        sample,fpr, tpr, auc, auc_thresh = dat   
        plt.plot(fpr, tpr, color=sample2col[sample],lw=2,)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate',size=16)
    plt.ylabel('True Positive Rate',size=16)
    plt.legend(handles=patches,fontsize=14)
    plt.savefig(os.path.join(outDirect,outFile))
    plt.show()
    return

def getAUCThresh(y,pred,thresh=0.1):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred)
    prTup_thresh = [(f,p) for (f,p) in zip(fpr,tpr) if f<=thresh]
    if len(prTup_thresh)<2:
        return -1,-1,-1
    fpr_thresh, tpr_thresh = list(zip(*prTup_thresh))
    auc_thresh = old_div(sklearn.metrics.auc(fpr_thresh,tpr_thresh),fpr_thresh[-1])
    return fpr_thresh, tpr_thresh, round(auc_thresh,3) 

def getAUC_EL_plotROC(y,pred,show=True,label=False):
    y_int = [int(i) for i in y]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_int, pred)
    auc = sklearn.metrics.auc(fpr, tpr)
    plotROC(fpr,tpr,auc,show=show,label=label)
    return fpr,tpr,auc

def plotROC(fpr,tpr,auc,show=True,label=False):
    if not label:
        label='ROC Curve'
    plt.plot(fpr, tpr,
         #lw=2,
             label='{} AUC: {}'.format(label,round(auc,3)))
    plt.plot([0, 1], [0, 1], color='black', 
             #lw=2, 
             linestyle='--')
    plt.xlabel('False Positive Rate',size=20)
    plt.ylabel('True Positive Rate',size=20)
    if show:
        plt.legend(loc=(0.001,1.01))
        plt.show()
    return

def plotROC_BA(fpr,tpr,auc):
    plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (AUC:%0.3f)' % (auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate',size=16)
    plt.ylabel('True Positive Rate',size=16)
    plt.legend()
    plt.show()
    return

def fullDF2AUCproc(df,by='AlleleList_ID',datType='BA',plot_ROC=False):
    summaryDict = {}
    for group in df.groupby(by=by):
        allele = group[0]
        groupDF = group[1]
        peptNum = len(groupDF)
        y,pred = getTargetPred(groupDF)
        if peptNum<2:
            continue
        PCC = scipy.stats.pearsonr(y,pred)[0]
        PCC = round(PCC,3)
        if datType=='BA':
            fpr,tpr,auc = getAUC_BA(y,pred)
            auc = round(auc,3)
            summaryDict[allele] = {'AUC':auc,'PCC':PCC,'Pept#':peptNum} 
        elif datType=='EL':
            fpr,tpr,auc = getAUC_EL(y,pred)
            auc = round(auc,3)
            posNum = int(groupDF['Measure'].values.sum())
            PPV = getPPV_pos(y,pred,posNum)
            PPV = round(PPV,3)
            fprT,tprT,aucT = getAUCThresh(y,pred,thresh=0.1)
            aucT = round(aucT,3)
            summaryDict[allele] = {'AUC':auc,'PCC':PCC,'Pept#':peptNum,
                                  'PosLig#':posNum,'PPV':PPV,'AUCthresh':aucT} 
        if plot_ROC:
            plotROC_BA(fpr,tpr,auc)
    return summaryDict

class NNalignOutput_BA(object):
    
    targCol = 'Measure'
    predCol = 'Prediction'
    peptNumThresh = 0
    
    def __init__(self,df,allele=False,plotROC =False,
                 targCol=False,predCol=False,peptNumThresh=False):
        if allele=='NA':
            self.allele = allele
        else:
            self.allele = str(df['MHC'].values[0])
        self.df = df
        self.numPept = len(self.df)
        self.updateDefaults(targCol,predCol,peptNumThresh)
        self.getStats()
        if plotROC:
            self.plotROC(self)
            
    def getStats(self):
        if self.numPept>self.peptNumThresh:
            self.y, self.pred = self.getTargetPred()
            self.PCC = self.getPCC()
            self.fpr, self.tpr, self.auc = self.getAUC()
            self.summaryStats = {'AUC':self.auc,'PCC':self.PCC,
                                 'Pept#':self.numPept,'Allele':self.allele}
        else:
            self.summaryStats = {'AUC':0,'PCC':0,
                                 'Pept#':self.numPept,'Allele':self.allele}        
        
    def getTargetPred(self):
        return self.df[self.targCol].apply(float), self.df[self.predCol].apply(float)

    def affTrans(self,affinityNm):
        #Apply normalization function as defined by Morten in some early publication 
        #- find that publication
        if affinityNm < 1:
            return 1.0
        elif affinityNm > 50000:
            return 0.0
        else:
            return 1 - old_div(np.log10(affinityNm),np.log10(50000))
    
    def getAUC(self, bind_thresh = 500): #500nm IC50 threshold for binding, transformed to range [0,1]
        bind_thresh_log = self.affTrans(bind_thresh)
        y_bool = [1 if i>bind_thresh_log else 0 for i in self.y]
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_bool, self.pred)
        auc = sklearn.metrics.auc(fpr, tpr)
        return fpr,tpr,round(auc,3)
    
    def getPCC(self):
        PCC = scipy.stats.pearsonr(self.y,self.pred)[0]
        return round(PCC,3)
    
    def updateDefaults(self,targCol,predCol,peptNumThresh):
        self.targCol=targCol if targCol else 'Measure'
        self.predCol=predCol if predCol else 'Prediction'
        self.peptNumThresh=peptNumThresh if peptNumThresh else 0
    
    def plotROC(self):
        plt.plot(self.fpr, self.tpr, color='darkorange',
         lw=2, label='ROC curve (AUC:%0.3f)' % (self.auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate',size=16)
        plt.ylabel('True Positive Rate',size=16)
        plt.legend()
        plt.show()
            
class NNalignOutput_EL(NNalignOutput_BA):

        def __init__(self,df,allele=False,plotROC=False,
                 targCol=False,predCol=False,peptNumThresh=False):
            if allele=='NA':
                self.allele = allele
            else:
                self.allele = str(df['MHC'].values[0])
            self.df = df
            self.numPept = len(self.df)
            self.updateDefaults(targCol,predCol,peptNumThresh)
            self.posNum = int(sum(self.df[self.targCol].apply(int).values))
            self.getStats()
            
            if plotROC:
                self.plotROC()
        
        def getStats(self):
            if self.numPept > self.peptNumThresh and not self.posNum < self.peptNumThresh:
                self.y, self.pred = self.getTargetPred()
                self.PCC = self.getPCC()
                self.fpr, self.tpr, self.auc = self.getAUC(self.y,self.pred)
                self.fprT,self.tprT,self.aucT = self.getAUCThresh()
                self.PPV = self.getPPV_pos()
                self.PPV9 = self.getPPV_pos(thresh=0.9)
                self.PPV5 = self.getPPV_pos(thresh=0.5)
                self.summaryStats = {'AUC':self.auc,'AUCthresh':self.aucT,'PCC':self.PCC,
                                    'Pept#':self.numPept,'PosLig#':self.posNum,
                                    'PPV':self.PPV,'PPV0.9':self.PPV9,
                                    'PPV0.5':self.PPV5,
                                    'Allele':self.allele} 
            else:
                self.summaryStats = {'AUC':0,'PCC':0,'Pept#':self.numPept,
                                  'PosLig#':self.posNum,'PPV':0,'PPVthresh':0,'AUCthresh':0,
                                    'Allele':self.allele}
        def getAUC(self,y,pred):
            y_int = [int(i) for i in y]
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_int, pred)
            try:
                auc = sklearn.metrics.auc(fpr, tpr)
            except ValueError:
                auc = -1.000
            return fpr,tpr,round(auc,3)
        
        def getPPV_pos(self,thresh=1.0):
            if self.posNum==0:
                return -1.000
            zipPred = list(zip(self.pred,self.y))
            zipPred.sort(reverse=True)
            numPred = len(zipPred)
            posNumThresh = int(self.posNum*thresh)
            if posNumThresh==0:
            	return -1
            pred_top,y_top = list(zip(*zipPred[:posNumThresh]))
            top_pos = sum([int(i) for i in y_top])
            PPV = float(top_pos)/int(self.posNum*thresh)
            return round(PPV,3)
        
        def getAUCThresh(self,thresh=0.1):
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.y, self.pred)
            prTup_thresh = [(f,p) for (f,p) in zip(fpr,tpr) if f<=thresh]
            if len(prTup_thresh)<2:
                return -1,-1,-1
            fpr_thresh, tpr_thresh = list(zip(*prTup_thresh))
            auc_thresh = old_div(sklearn.metrics.auc(fpr_thresh,tpr_thresh),fpr_thresh[-1])
            return fpr_thresh, tpr_thresh, round(auc_thresh,3) 

        
        #def getAUCThresh(self,thresh=0.1):
        #    fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.y, self.pred)
        #    prTup = list(zip(fpr,tpr))
        #    idx = 0
        #    prTup_thresh = [pr for pr in prTup if pr[0]>=thresh]
        #    if len(prTup_thresh)<2:
        #    	return -1,-1,-1
        #    #for i,pr in enumerate(prTup):
        #    #    if pr[0]>=thresh:
        #    #        idx=i
        #    #        break
        #    #fpr_thresh, tpr_thresh = list(zip(*prTup[:idx]))
        #    fpr_thresh, tpr_thresh = list(zip(*prTup_thresh))
        #    auc_thresh = old_div(sklearn.metrics.auc(fpr_thresh,tpr_thresh),fpr_thresh[idx-1])
        #    return fpr_thresh, tpr_thresh, round(auc_thresh,3) 
        
        #def getAUCThresh(self,thresh=0.1):
        #    thresh = int(100*thresh)
        #    zipPred = zip(self.pred,self.y)
        #    zipPred.sort(reverse=True)
        #    numPred = len(zipPred)
        #    pred01,y01 = zip(*zipPred[:numPred/thresh])
        #    fprT,tprT,aucT = self.getAUC(y01,pred01)
        #    return fprT,tprT,round(aucT,3)
        
class NNalignOutput_splitter(object):
    
    allele = 'NA'
    datType = 'BA'
    
    def __init__(self,df_master,groupby,datType,targCol=False,predCol=False,peptNumThresh=False):
        self.peptNumThresh = peptNumThresh
        self.updateDefaults(targCol,predCol,peptNumThresh)
        df_master = self.inputTypeCast(df_master)
        self.objectDict = self.groupSplitClass(df_master,groupby,datType)
        self.summaryDict = {key:val.summaryStats for key,val in list(self.objectDict.items())}
        self.summaryDF = self.sumDict2DF(dropZero=True)
        self.summaryDFresetIndex(groupby)
        self.summaryTypeCast()
        
    def groupSplitClass(self,df,groupby,datType):
        objectDict = {}
        for group in df.groupby(groupby):
            split = group[0]
            df_split = group[1]
            if groupby=='MHC':
                self.allele=split
            if datType=='BA':
                objectDict[split] = NNalignOutput_BA(df_split,allele=self.allele,targCol=self.targCol,predCol=self.predCol,peptNumThresh=self.peptNumThresh)
            elif datType=='EL':
                objectDict[split] = NNalignOutput_EL(df_split,allele=self.allele,targCol=self.targCol,predCol=self.predCol,peptNumThresh=self.peptNumThresh)
        return objectDict
    
    def sumDict2DF(self,dropZero=False):
        summaryDF = pd.DataFrame.from_dict(self.summaryDict).T
        summaryDF = summaryDF.dropna()
        summaryDF = summaryDF.sort_index()
        if dropZero:
            try:
                summaryDF = summaryDF[summaryDF['AUC']>0]
            except KeyError:
                summaryDF = summaryDF[summaryDF['AUCthresh']>0]
        return summaryDF
        
    def writeSplit(self,outDir,malCond=False,onlyPos=False,prefix = 'splitOut',col=False):
        for key, val in list(self.objectDict.items()):
            val = val.df
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            if malCond:
                if key[0]==key[1]:#If dataset ID equals MHC, then SAL
                    continue
            if onlyPos:
                val = val[val[self.targCol]==1.0]
            if type(key) is tuple:
                key = '__'.join(key)
            outDirFile = os.path.join(outDir, '%s__%s.txt'%(prefix,key))
            if col:
                val[col].to_csv(outDirFile, header=None, index=False)
            else:
                val.to_csv(outDirFile, header=None, index=False)
    
    def inputTypeCast(self,df_master):
        df_master[self.targCol] = df_master[self.targCol].apply(float)
        df_master[self.predCol] = df_master[self.predCol].apply(float)
        return df_master

    def updateDefaults(self,targCol,predCol,peptNumThresh):
        self.targCol=targCol if targCol else 'Measure'
        self.predCol=predCol if predCol else 'Prediction'
        self.peptNumThresh=peptNumThresh if peptNumThresh else 0

    def summaryTypeCast(self):
        self.summaryDF['AUC'] = self.summaryDF['AUC'].apply(lambda x: x if x==-1 else float(x))
        self.summaryDF['Pept#'] = self.summaryDF['Pept#'].apply(lambda x: x if x==-1 else int(x))
        try:
            self.summaryDF['AUCthresh'] = self.summaryDF['AUCthresh'].apply(lambda x: x if x==-1 else float(x))
        except KeyError:
            pass
        try:
            self.summaryDF['PPV'] = self.summaryDF['PPV'].apply(lambda x: x if x==-1 else float(x))
            self.summaryDF['PPV0.9'] = self.summaryDF['PPV0.9'].apply(lambda x: x if x==-1 else float(x))
            self.summaryDF['PPV0.5'] = self.summaryDF['PPV0.5'].apply(lambda x: x if x==-1 else float(x))
        except KeyError:
            pass

        #self.summaryDF['AUC'] = list(map(float, self.summaryDF['AUC'].values))
        #self.summaryDF['Pept#'] = list(map(float, self.summaryDF['Pept#'].values))
        #try:
        #    self.summaryDF['AUCthresh'] = list(map(float, self.summaryDF['AUCthresh'].values))
        #except KeyError:
        #    pass
    def summaryDFresetIndex(self,groupby):
        if len(groupby)==1:
            self.summaryDF = self.summaryDF.reset_index().rename(columns={"index":groupby[0]})
        else:
            self.summaryDF = self.summaryDF.reset_index().rename(columns={"level_{}".format(i):s for i,s in enumerate(groupby)})

        
def NetmhcpanOutput2DF(datDir, inFile):
    header = ['HLA','Peptide','Core','dummy','Of','Gp','Gl','Ip','Il','Icore','Identity','Score Aff(nM)', '%Rank','dummy2','BindLevel']
    return pd.read_csv(os.path.join(datDir,inFile),sep="\s+",engine='python',quotechar='#',header=None,names=header)#read the file, separate columns, define heads

def NetMHCIIpanOutput2DF(datDir, inFile,skiprows=11,skipfooter=3,header=False):
    if not header:
        header = ['Seq','Allele','Peptide','Identity','Pos','Core','Core_Rel','1-log50k(aff)','Affinity(nM)','%Rank','Exp_Bind','BindingLevel']
    return pd.read_csv(os.path.join(datDir,inFile),sep="\s+",engine='python',quotechar='#',header=None,names=header,skiprows=skiprows,skipfooter=skipfooter)#read the file, separate columns, define heads

