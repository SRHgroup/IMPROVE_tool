from __future__ import print_function
from builtins import map
from builtins import zip
from builtins import str
import matplotlib.patches as mpatches
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
import FrankAnalysis
from scipy.stats import mannwhitneyu

def printStats(df,x,y):
    maxLen = max([len(x[0]) for x in df.groupby(x)])
    if type(x) is list:
        print('-'.join(x).ljust(maxLen+2),"Mean %s"%(y).ljust(1+len(y)),"Median %s"%(y).ljust(7+len(y)))
    else:
        print(x.ljust(maxLen+2),"Mean %s"%(y).ljust(1+len(y)),"Median %s"%(y).ljust(7+len(y)))
    #print "%s\tMean %s\tMedian %s"%(x,y,y)
    for group in df.groupby(x):
        if type(group[0]) is tuple:
            print('-'.join(group[0]).ljust(maxLen+2), str(round(group[1][y].mean(),3)).ljust(6+len(y)), str(round(group[1][y].median(),3)).ljust(7+len(y)))
        else:
            print(group[0].ljust(maxLen+2), str(round(group[1][y].mean(),3)).ljust(6+len(y)), str(round(group[1][y].median(),3)).ljust(7+len(y)))    


        
def makeJitterBoxplot_manuscript(df,y='AUC',x='Model',save=False,pStats=False,legNCol=2,ylabel='',alpha=1.0,show=True):
    
    ax = sns.stripplot(x=x,y=y,data=df,jitter=True,alpha=alpha)
    sns.boxplot(x=x,y=y,data=df,color='white',fliersize=0.1)

    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(y)
    plt.xlabel('')
    plt.tick_params(axis='both', which='major')

    if save:
        plt.savefig(save, bbox_inches='tight')
    if pStats:
        printStats(df,x,y)
    if show:
        plt.show()
    else:
        return ax

def makeJitterBoxplotHue_manuscript(df,y='AUC',x='Model',hue=False,
                save=False,ylim=False,pStats=False,
                ylabel='',xlabel='',show=True,
                alpha=1.0,jitter=True,
                legendSub=True,legend=False,Xrot=False,
                yAxLog=False,dpi=300,
                **legendKW):
    if hue:
        ax = sns.stripplot(x=x,y=y,hue=hue,data=df,dodge=True,jitter=jitter,alpha=alpha)
        sns.boxplot(x=x,y=y,hue=hue,data=df,color='white')
        hueList = list(set(df[hue].values))
    else:
        ax = sns.stripplot(x=x,y=y,data=df,dodge=True,jitter=jitter,alpha=alpha)
        sns.boxplot(x=x,y=y,data=df,color='white')
        hueList = list(set(df[x].values))
        
    if legendSub:
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles[len(hueList):],labels[len(hueList):],
                  **legendKW)
        legend.get_frame().set_linewidth(2)
    if ylabel:
        plt.ylabel(ylabel)
    else:  
        plt.ylabel(y)
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('')
    if ylim:
        plt.ylim(ylim)
    plt.tick_params(axis='both', which='major')#, labelsize=22)

    if Xrot:
        plt.xticks(rotation=Xrot)
    if pStats:
        if hue:
            printStats(df,[x,hue],y)
        else:
            printStats(df,[x],y)
    if yAxLog:
        ax.set(yscale="log")            
    if save:
        plt.savefig(save, bbox_inches='tight',dpi=dpi)        
    if show:
        plt.show()
    else:
        return ax

def makeJitterBoxplotHue_manuscript_grayscale(df,y='AUC',x='Model',hue=False,
                                            hues = ['MA-Model','MAC-Model'],markers = ['.','P'],
                save=False,ylim=False,pStats=False,
                ylabel='',xlabel='',show=True,
                alpha=1.0,jitter=True,
                legendSub=True,legend=False,Xrot=False,
                yAxLog=False,dpi=300,
                **legendKW):
    plt.clf()
    ax = sns.stripplot(x=x,y=y,hue=hue,data=df,dodge=True,jitter=jitter,alpha=alpha,color='black',s=7)
    #ax = sns.stripplot(x=x,y=y,hue=hue,data=df,dodge=True,jitter=jitter,alpha=alpha)
    sns.boxplot(x=x,y=y,hue=hue,data=df,color='grey')
    hueList = list(set(df[hue].values))
    
        
    if legendSub:
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles[:len(hueList)],labels[:len(hueList)],
                  **legendKW)
        legend.get_frame().set_linewidth(2)
    if ylabel:
        plt.ylabel(ylabel)
    else:  
        plt.ylabel(y)
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('')
    if ylim:
        plt.ylim(ylim)
    plt.tick_params(axis='both', which='major')#, labelsize=22)

    if Xrot:
        plt.xticks(rotation=Xrot)
    if yAxLog:
        ax.set(yscale="log")            
    if save:
        plt.savefig(save, bbox_inches='tight',dpi=dpi)        
    if show:
        plt.show()
    else:
        return ax
    
def makeJitterBoxplot_manuscriptKW(df,y='AUC',x='Model',hue=False,
                save=False,ylim=False,pStats=False,
                ylabel='',xlabel='',show=True,
                alpha=1.0,jitter=True,
                legendSub=False,xrot=False,
                yAxLog=False,dpi=300,
                **legendKW):
    if hue:
        ax = sns.stripplot(x=x,y=y,hue=hue,data=df,dodge=True,jitter=jitter,alpha=alpha)
        sns.boxplot(x=x,y=y,hue=hue,data=df,color='white')
        hueList = list(set(df[hue].values))
    else:
        ax = sns.stripplot(x=x,y=y,data=df,dodge=True,jitter=jitter,alpha=alpha)
        sns.boxplot(x=x,y=y,data=df,color='white')
        hueList = list(set(df[x].values))
        
    if legendSub:
        hueList = list(set(df[hue].values))
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles[len(hueList):],labels[len(hueList):],
                  **legendKW)
        legend.get_frame().set_linewidth(2)
        #legend.get_frame().set_edgecolor("black")
    if ylabel:
        plt.ylabel(ylabel)
    else:  
        plt.ylabel(y)
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('')
    if ylim:
        plt.ylim(ylim)
    plt.tick_params(axis='both', which='major')#, labelsize=22)
    if xrot:
        plt.xticks(rotation=xrot)
    if yAxLog:
        ax.set(yscale="log")
    if save:
        plt.savefig(save, bbox_inches='tight',dpi=dpi)
    if pStats and not hue:
        printStats(df,[x],y)
    elif pStats and hue:
        printStats(df,[x,hue],y)
    if show:
        plt.show()
    else:
        return ax

def makeJitterBoxplotHue(df,order,colors=['red','skyblue'],y='AUC',x='Model',hue='Allele',save=False,ylim=False,pStats=False,legNCol=2,ylabel='',alpha=1.0):
    ax = sns.stripplot(x=x,y=y,hue=hue,data=df,jitter=True,alpha=alpha,dodge=True)
    
    hueList = list(set(df[hue].values))
    pal2 = dict(list(zip(hueList,['w']*len(hueList))))
    sns.boxplot(x=x,y=y,hue=hue,data=df,palette=pal2)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[len(hueList):],labels[len(hueList):],ncol=legNCol,loc=(0.01,1.05))
    if ylabel:
        plt.ylabel(ylabel)
    else:  
        plt.ylabel(y)
    plt.xlabel('')
    if ylim:
        plt.ylim(ylim)
    plt.tick_params(axis='both', which='major')#, labelsize=22)

    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()

    if pStats:
        printStats(df,[x,hue],y)
        
################ Compare distributions

def groupKDEPlot(df,groupCol='Loci',distCol = 'PPV0.9',show=True,saveFig=False,legendLoc=[1.02,0.4],clip=None,pStats=True,**pltKW):
    plt_kws = {'linewidth':4.0}
    plt_kws.update(pltKW)
    for group, dfGroup in df.groupby(groupCol):
        sns.kdeplot(dfGroup[distCol],label=group,clip=clip,**plt_kws)
        if pStats:
            print("{}-{} Mean: {} Median: {}".format(group,distCol,round(dfGroup[distCol].mean(),3),round(dfGroup[distCol].median(),3)))
    plt.legend(loc=legendLoc)
    if saveFig:
        plt.savefig(saveFig)    
    if show:
        plt.show()

def groupHistPlot(df,groupCol='Loci',distCol = 'PPV0.9',show=True,saveFig=False,legendLoc=[1.02,0.4]):
    for group, dfGroup in df.groupby(groupCol):
        sns.distplot(dfGroup[distCol],label=group,kde=False)
    plt.legend(loc=legendLoc)
    if saveFig:
        plt.savefig(saveFig)
    if show:
        plt.show()

def groupDistPlot(df,groupCol='Loci',distCol = 'PPV0.9',show=True,saveFig=False,kde=True,hist=False,legend=True,legendLoc=[1.02,0.4],**kdeKW):
    kde_kws={'linewidth':4.0}
    kde_kws.update(kdeKW)
    print(kde_kws)
    for group, dfGroup in sorted(df.groupby(groupCol)):
        if kde:
            sns.distplot(dfGroup[distCol],label=group,hist=False,kde_kws=kde_kws)
        if hist:
            sns.distplot(dfGroup[distCol],label=group,kde=False)
    if legend:
        plt.legend(loc=legendLoc)
    if saveFig:
        plt.savefig(saveFig)
    if show:
        plt.show()

#### Basic Scatter plot with correlation printing
        
def scatterCorr(df,x,y,hue=False,printing=True,show=True,save=False,dpi=600,corrAnnot=False,size=100,corrAnnot_spearman=True):
    dfOut = df.copy(deep=True)
    if hue:
        dfOut = dfOut[[x,y,hue]]
    else:
        dfOut = dfOut[[x,y]]
    preDropLen = len(dfOut)
    dfOut = dfOut.dropna()
    postDropLen = len(dfOut)
    droppedRows = preDropLen-postDropLen
    if  droppedRows != 0:
        print("Dropped {} rows with NA values".format(droppedRows))
     
    if hue:
        p1 = sns.scatterplot(dfOut[x],dfOut[y],hue=dfOut[hue],s=size)
        plt.legend(loc=(1.02,0.25))
    else:
        p1 = sns.scatterplot(dfOut[x],dfOut[y],s=size)
    spearmanCorr = scipy.stats.spearmanr(dfOut[x],dfOut[y])[0]
    pearsonCorr = scipy.stats.pearsonr(dfOut[x],dfOut[y])[0]
    if printing:
        print("Spearmann Corr: {:.4f}".format(spearmanCorr))
        print("Pearson Corr: {:.4f}".format(pearsonCorr))
    if corrAnnot:
        if corrAnnot_spearman:
            annotString = "Spearmanr: {}".format(round(spearmanCorr,3))            
        else:
            annotString = "Pearsonr: {}".format(round(pearsonCorr,3))

        p1.text(corrAnnot[0], corrAnnot[1], annotString, horizontalalignment='left', size='medium', color='black', weight='normal')

    if save:
        plt.savefig(save,dpi=dpi)
    if show:
        plt.show()
    


def regPlotAnnot(df,x,y,group,dx=0.2,dy=0,fit_reg=True,printing=True,annot=False,show=True,save=False,dpi=600):
    dfOut = df.copy(deep=True)
    preDropLen = len(dfOut)
    dfOut = dfOut.dropna()
    postDropLen = len(dfOut)
    droppedRows = preDropLen-postDropLen
    if  droppedRows != 0:
        print("Dropped {} rows with NA values".format(droppedRows))
     
    p1 = sns.regplot(data=dfOut,x=x,y=y,fit_reg=fit_reg)
    if annot:
        for line in range(0,df.shape[0]):
            p1.text(df[x][line]+dx, df[y][line]+dy, df[group][line], horizontalalignment='left', size='medium', color='black', weight='semibold')
    if save:
        plt.savefig(save,dpi=dpi)
    if show:
        plt.show()
    if printing:
        print("Spearmann Corr: {}".format(scipy.stats.spearmanr(dfOut[x],dfOut[y])[0]))
        print("Pearson Corr: {}".format(scipy.stats.pearsonr(dfOut[x],dfOut[y])[0]))

def regLinearPlot(df,x,y,dx=0.2,dy=0,fit_reg=True,printing=True,annot=False,show=True,save=False,dpi=600):
    dfOut = df.copy(deep=True)
    preDropLen = len(dfOut)
    dfOut = dfOut.dropna()
    postDropLen = len(dfOut)
    droppedRows = preDropLen-postDropLen
    if  droppedRows != 0:
        print("Dropped {} rows with NA values".format(droppedRows))

    slope, intercept, r_value, pv, se = scipy.stats.linregress(dfOut[x], dfOut[y])
    sns.regplot(data=dfOut,x=x,y=y,line_kws={'label':'$y=%3.4s*x+%3.4s$ r^2: %3.4s'%(slope, intercept,r_value)})
    plt.legend(loc=(0.001,1.01))
        
    if save:
        plt.savefig(save,dpi=dpi)
    if show:
        plt.show()
    if printing:
        print("Spearmann Corr: {}".format(scipy.stats.spearmanr(dfOut[x],dfOut[y])[0]))
        print("Pearson Corr: {}".format(scipy.stats.pearsonr(dfOut[x],dfOut[y])[0]))
        
############ Count instances in a column

def dfColCountBarplot(df,countCol='Source',orient='h'):
    bars = df.groupby(countCol).apply(len)
    bars_df = pd.DataFrame(bars).T
    sns.barplot(data=bars_df,orient=orient)
    plt.show()
    return bars_df

        
#################Plotting kmer prediction score profiles

def kmerDir2measPredDict(inDir,emptyFiles,cond='log_eval',model='nnalign'):
    measPredDict = {}
    for filename in os.listdir(inDir):
        if filename in emptyFiles:#Skip over empty files
            continue
        if cond in filename:
            #print filename
            measPredDict[filename] = FrankAnalysis.kmerFile2predMeas(inDir,filename,model=model)
    return measPredDict

def measPredScatter(measPred,label='',legend=False,show=True):
    meas, pred = list(zip(*measPred))
    pred = list(map(float, pred))
    meas = list(map(float,meas))
    epiIDX = [(i,pred[i])for i,m in enumerate(meas) if m==1.0]
    i,v = list(zip(*epiIDX))
    #plt.scatter(range(len(pred)),pred,c=pred,cmap='Greys')
    plt.plot(pred,label=label,zorder=0)
    #plt.plot(range(len(meas)),meas)
    plt.scatter(i,v,c='r',marker='X',s=200,zorder=1)
    plt.xlabel('Kmer')#,fontsize=16)
    plt.ylabel('Prediction Score')#,fontsize=16)
    if legend:
        plt.legend()
    if show:
        plt.show()

def kmerFile2measPredScatter(inDir,filename,label='',legend=False,show=True,model='nnalign'):
    measPredScatter(FrankAnalysis.kmerFile2predMeas(inDir,filename,model=model),label=label,legend=legend,show=show)

def plotKmerProfile_selectFiles(dir1,fileCond='log_eval',fileCond2=False):
    filelist = [filename for filename in os.listdir(dir1) if fileCond in filename]
    if fileCond2:
        filelist = [filelist for filename in filelist if fileCond2 in filename]
    for filename in filelist:
        print(filename)
        kmerFile2measPredScatter(dir1,filename,label='EL',show=False)
        kmerFile2measPredScatter(dir2,filename,label='EL_proc',legend=True)    

def compareKmerProfile(dir1,dir2,fileCond='log_eval',fileCond2=False):
    filelist = [filename for filename in os.listdir(dir1) if fileCond in filename]
    #if fileCond2.any():
    #    filelist = [filelist for filename in filelist for fileCond in fileCond2 if fileCond in filename]
    for filename in filelist:
        print(filename)
        kmerFile2measPredScatter(dir1,filename,label='EL',show=False)
        kmerFile2measPredScatter(dir2,filename,label='EL_proc',legend=True)



def recistSwarmPlot(patVarCount,saveFig=False,y='Count',yPlus=5,h=5,ylabel='Tumor Mutational Burden'):
    testMapper = {'CR':"SD/PR/CR",
                 'SD':"SD/PR/CR",
                  'PR':"SD/PR/CR",
                  'PD':"PD"
                 }
    patVarCount['testCat'] = patVarCount['RECIST'].map(testMapper)

    patVarCount = patVarCount[patVarCount['RECIST'].notna()]
    patVarCount = patVarCount.sort_values('testCat')
    
    patBetter = patVarCount[patVarCount['testCat']=="SD/PR/CR"][y].values
    patWorse = patVarCount[patVarCount['testCat']=="PD"][y].values

    mannWhitney = mannwhitneyu(patBetter,patWorse,alternative='Greater')
    pval = round(mannWhitney[1],3)
    print(mannWhitney)

    plt.figure(figsize=(9,6))

    sns.swarmplot(data=patVarCount,x='testCat',y=y,hue='RECIST',size=15)
    sns.boxplot(data=patVarCount,x='testCat',y=y,color='white')

    plt.legend(loc=(1.0,0.3))
    plt.xlabel('Patient Outcome')
    plt.ylabel(ylabel)


    # statistical annotation
    x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, col = patVarCount[y].max() + yPlus, 'k'

    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "p = {}".format(pval), ha='center', va='bottom', color=col,fontsize=26)

    plt.ylim(0,y+3*h)
    if saveFig:
        plt.savefig(saveFig,dpi=600)
    plt.show()