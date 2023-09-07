import os
import pandas as pd
import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import partitionTools
import itertools

def getGroupTopX(df,groupCol='Patient',predCol='CombFeat_idx',X=50,rank=True):
    if rank:
        return df.groupby(groupCol).apply(lambda dfG: dfG.nsmallest(X,predCol)).reset_index(drop=True)
    else:
        return df.groupby(groupCol).apply(lambda dfG: dfG.nlargest(X,predCol)).reset_index(drop=True)

def rankByFeaturePerPatient(df,feature='EL',ascending=True,groupCol='Patient'):
    dfList = []
    for patient, dfG in df.groupby(groupCol):
        dfOut = dfG.copy(deep=True)
        dfOut = dfOut.sort_values(feature,ascending=ascending)
        dfOut['{}_idx'.format(feature)] = list(range(1,len(dfOut)+1))
        dfList.append(dfOut)
    return pd.concat(dfList)

def rankMultiFeaturePerPatient(df,featureDict):
    dfOut = df.copy(deep=True)
    for feature,ascending in featureDict.items():
        #print(feature)
        dfOut = rankByFeaturePerPatient(dfOut,feature=feature,ascending=ascending)
    return dfOut

def evalDFonMultiFeatureWrapper(df,features,featureDict,aggFunc = scipy.stats.mstats.gmean):
    dfOut = rankMultiFeaturePerPatient(df,featureDict)
    dfOut = genCombRankFeats(dfOut,features,aggFunc=aggFunc)
    return rankByFeaturePerPatient(dfOut,feature='CombFeat',ascending=True)

def printFeatureYields(df,feature,rank=True,X=50,relative=True):
    totalRespNum = df['Target'].sum()
    df_top50 = getGroupTopX(df,groupCol='Patient',predCol=feature,X=X)
    top50RespNum = df_top50['Target'].sum()
    if relative:
        return top50RespNum/totalRespNum
    else:
        return top50RespNum

def genCombRankFeats(df,featureComb,aggFunc = np.mean,X=50):
    df_comb = df.copy(deep=True)
    df_comb['CombFeat'] = df_comb[list(featureComb)].agg(aggFunc,axis=1)
    df_comb = rankByFeaturePerPatient(df_comb,feature='CombFeat',ascending=True)
    return df_comb 

def getPatientHits(df,X=50,predCol='CombFeat_idx'):
    df_top50 = getGroupTopX(df,groupCol='Patient',predCol=predCol,X=X)
    numHits_top50 = df_top50.groupby('Patient')['Target'].sum().reset_index()
    patHits_top50 = sum(numHits_top50['Target']>0)
    return patHits_top50

def getTopXTargetCount(df,groupCol='Patient',targCol='Target',colName = 'Top50Hits'):
    dfTop50 = df.groupby(groupCol).apply(lambda dfG: dfG[targCol].sum()).reset_index().rename(columns={0:colName})
    yields = getHitRatio(dfTop50,colName=colName)
    return dfTop50,yields

def getHitRatio(dfTop50, colName = 'Top50Hits', printYields = True):
    numPatients = len(dfTop50)
    hitPatients = sum(dfTop50[colName]>0)
    medianEpiHit = dfTop50[colName].median()
    meanEpiHit = dfTop50[colName].mean()
    yields = (numPatients,hitPatients,hitPatients/numPatients,meanEpiHit,medianEpiHit)
    if printYields:
        print(yields)
    return yields

def topXPatientEpiYield_wrapper(df):
    df_top50 = getGroupTopX(df)
    return getTopXTargetCount(df_top50)

def combRankFeats(df,featureComb,aggFunc = np.mean,X=50,relative=True):
    df_comb = df.copy(deep=True)
    df_comb['CombFeat'] = df_comb[list(featureComb)].agg(aggFunc,axis=1)
    df_comb = rankByFeaturePerPatient(df_comb,feature='CombFeat',ascending=True)
    patientHits = getPatientHits(df_comb,X=X)
    epitopeYields = printFeatureYields(df_comb,'CombFeat_idx',rank=True,X=X,relative=relative)
    return patientHits,epitopeYields

def testFeatureCombinations(df,numComb,aggFunc = np.mean,X=50):
    rankCols = [col for col in df.columns if col.endswith('_idx')]
    #print(rankCols)
    dfOut_comb = df.copy(deep=True)
    featureCombResultList = []
    for featureComb in itertools.combinations(rankCols, numComb):
        patientHits,top50Yield = combRankFeats(dfOut_comb,featureComb,aggFunc=aggFunc,X=X)
        featureCombResultList.append((featureComb,top50Yield,patientHits))
    bestFeats = sorted(featureCombResultList,key=lambda x: x[1:],reverse=True)
    return bestFeats

def forwardFeatureSelectionCV(df,featureDict,maxFeat=5,sortCol='MeanEpiHits',modelType='NN',**modelKW):
    features_start = featureDict.keys()
    features_select = []
    features_left = features_start
    scoreDict_master = {}
    bestFeatureScore_dfList = []
    for i in range(1,maxFeat+1):
        yieldDict_list = []
        #print(i)
        for feat in features_left:
            features_current = features_select + [feat]
            dfOut,modelDict = runCrossValidationTraining(df,features_current,featureDict,modelType=modelType,**modelKW)
            yieldDict = yieldsWrapperStats(dfOut,hue=True,targCol='Target',predCol='classPred',plot=False,rank=False,Print=False)
            yieldDict['AUC'] = computeROCAUC(dfOut)
            yieldDict['PPV'] = getPPV(dfOut)
            #yieldDict['Feat'] = feat
            yieldDict['Features_current'] = features_current
            yieldDict_list.append(yieldDict)
        yieldDF = pd.DataFrame(yieldDict_list)
        yieldDF['Features'] = features_left
        yieldDF['FeatNumber'] = i
        yieldDF = yieldDF.sort_values(sortCol,ascending=False)
        yieldDF_top = yieldDF.iloc[0,:]
        features_select = yieldDF_top['Features_current']
        features_left = [feat for feat in features_start if not feat in features_select]
        bestFeatureScore_dfList.append(yieldDF_top)
    return pd.DataFrame(bestFeatureScore_dfList)


def testFeatureCombinations_greedy(df,numComb,maxFeat=15,aggFunc = np.mean,X=50):
    rankCols = [col for col in df.columns if col.endswith('_idx')]
    #print(rankCols)
    features_start = rankCols
    features_select = []
    features_left = features_start
    
    dfOut_comb = df.copy(deep=True)
    bestFeats = []
    for i in range(1,maxFeat+1):
        bestFeats_iter = []
        for feat in features_left:
            features_current = features_select + [feat]
            #print(features_current)
            patientHits,top50Yield = combRankFeats(dfOut_comb,features_current,aggFunc=aggFunc,X=X)
            bestFeats_iter.append((features_current,top50Yield,patientHits))
        bestFeats_iter = sorted(bestFeats_iter,key=lambda x: x[1:],reverse=True)
        #print(bestFeats_iter)
        features_select = bestFeats_iter[0][0]
        features_left = [feat for feat in features_start if not feat in features_select]
        bestFeats.append(bestFeats_iter[0])
    bestFeats = sorted(bestFeats,key=lambda x: x[1:],reverse=True)
    return bestFeats

def multiply(l):
    out = 1
    for ll in l:
        out*=ll
    return out

def getYieldDict(dfYields,targCol='Target',Print=True):
    targCol_top50 = '{}_top50'.format(targCol)
    yieldDict = {}
    
    yieldDict['Epitope-Total'] = int(dfYields[targCol].sum())
    yieldDict['Epitope-Hits'] = int(dfYields[targCol_top50].sum())
    if yieldDict['Epitope-Total']==0:#To get around zero division
        yieldDict['Epitope Yield%'] = 0
    else:
        yieldDict['Epitope Yield%'] = round(yieldDict['Epitope-Hits']/yieldDict['Epitope-Total']*100,1)
    #Of the patients that were hit, what is the median+mean number of epitopes
    yieldDict['MedianEpiHits'] = dfYields[dfYields[targCol_top50]>0][targCol_top50].median()
    yieldDict['MeanEpiHits'] = round(dfYields[dfYields[targCol_top50]>0][targCol_top50].mean(),2)
    
    yieldDict['Patient-Resp'] = sum(dfYields[targCol]>0)
    yieldDict['Patient-Hits'] = sum(dfYields[targCol_top50]>0)
    yieldDict['Patient-Total'] = len(dfYields)
    if yieldDict['Patient-Total']==0:
        yieldDict['Patient Hit%'] = 0
    else:
        yieldDict['Patient Hit%'] = round(yieldDict['Patient-Hits']/yieldDict['Patient-Total']*100,1)
    if Print:
        print(yieldDict)
    return yieldDict

def genYieldDF(yieldDicts,labels):
    yieldDF = pd.DataFrame(yieldDicts).T
    yieldDF['Features'] = labels
    return yieldDF

def getFeatureYields(df_comb,targCol='Target',predCol='CombFeat_idx',groupCol='Patient',X=50,rank=True):
    df_comb_top50 = getGroupTopX(df_comb,groupCol=groupCol,predCol=predCol,X=X,rank=rank)    
    numHits = df_comb.groupby(groupCol)[targCol].sum().reset_index()
    #numHits_top50 = df_comb_top50.groupby(groupCol)[targCol].sum().reset_index()
    #numHits_top50 = df_comb_top50.groupby(groupCol)[targCol].sum().reset_index()
    try:
        numHits_top50 = df_comb_top50.groupby(groupCol)[targCol].sum().reset_index()
    except KeyError:
        numHits_top50 = df_comb.groupby(groupCol)[targCol].sum().reset_index()
        #print(df_comb)
        #print(df_comb_top50)
        #sys.exit()

    numHits_merge = numHits.merge(numHits_top50,on=groupCol,suffixes=('','_top50'))
    numHits_merge['Yields'] = numHits_merge['{}_top50'.format(targCol)]/numHits_merge[targCol]*100
    numHits_merge[groupCol] = numHits_merge[groupCol].apply(lambda pat: str(pat))
    return numHits_merge

def patientYieldBarplot(dfYields,saveFig=False,targCol='Target',show=True):

    plt.figure(figsize=(18,8))
    ax = sns.barplot(data=dfYields,x='Patient',y='Yields',color=sns.color_palette()[0])

    targetDict_whole = dict(dfYields[['Patient',targCol]].values)
    targetDict_top50 = dict(dfYields[['Patient','{}_top50'.format(targCol)]].values)

    loc,labels = plt.xticks()
    labs = [label.get_text() for label in labels]

    for i,p in enumerate(ax.patches):
        totalNum = int(targetDict_whole[labs[i]])
        top50Num = int(targetDict_top50[labs[i]])
        ax.annotate("{}/{}".format(top50Num,totalNum),(p.get_x() * 1.0, p.get_height() + 1.0),fontsize=18)

    plt.ylabel("% Yields")
    plt.xticks(rotation=30)
    if not show:
        return 
    if saveFig:
        plt.savefig(saveFig)
    plt.show()

def patientYieldBarplotHue(dfYields,saveFig=False,targCol='Target',show=True):
    dfYields = dfYields[['Patient',targCol,'{}_top50'.format(targCol)]].melt(id_vars='Patient')
    plt.figure(figsize=(18,8))
    #print(dfYields.head())
    #plt.grid()    
    ax = sns.barplot(data=dfYields,x='Patient',y='value',hue='variable')

    plt.ylabel("Epitopes")
    plt.xticks(rotation=30)
    plt.yticks(range(0,int(dfYields['value'].max()),2))
    
    if not show:
        return 
    if saveFig:
        plt.savefig(saveFig,dpi=600)
    plt.show()    

def yieldsWrapperStats(df,targCol='Target',predCol='CombFeat_idx',rank=True,show=True,saveFig=False,hue=False,X=50,plot=True,Print=True):    
    dfYields = getFeatureYields(df,targCol=targCol,predCol=predCol,rank=rank,X=X)
    yieldDict = getYieldDict(dfYields,targCol=targCol,Print=Print)
    
    if plot:
	    if hue:
	        patientYieldBarplotHue(dfYields,saveFig = saveFig,targCol=targCol,show=show)
	    else:
	        patientYieldBarplot(dfYields,saveFig = saveFig,targCol=targCol,show=show)
    return yieldDict

def yieldsWrapper(df,features,targCol='Target',show=True,saveFig=False,hue=False,aggFunc = scipy.stats.mstats.gmean,X=50,plot=True):
    df_comb = genCombRankFeats(df,features,aggFunc = aggFunc)    
    return yieldsWrapperStats(df_comb,targCol=targCol,show=show,saveFig=saveFig,hue=hue,X=X,plot=plot)

def colorPatientXticks(hlaCountDict,colorDict,saveFig=False,show=True):
    idxs,labels = plt.xticks()
    for idx,label in zip(idxs,labels):
        patient = int(label.get_text())
        color = colorDict[hlaCountDict[patient]]
        label.set_color(color)
    if saveFig:
        plt.savefig(saveFig,dpi=600)
    if show:
        plt.show()

def summarizeGroupCumPlot(df,groupCol='Patient',valueCol='CombFeat_idx',func=min,ax=plt.gca()):
    df_stat = df.groupby(groupCol)[valueCol].apply(func).reset_index()
    sns.ecdfplot(data = df_stat, x=valueCol)

def cumulativeRankDistrib(df,x = 'CombFeat_idx',xtick=50,saveFig=False,showFig=True,label=None):
    sns.ecdfplot(data = df, x=x,label=label)
    m = df[x].max()
    plt.xticks(range(0,round(m,-2)+xtick,xtick))
    plt.yticks(np.linspace(0,1.0,11))
    plt.xlabel('Peptide Rank Per Patient')
    if saveFig:
        plt.savefig(saveFig.format('Cumulative'),dpi=600)
    if showFig:
	    plt.show()

def cumulativePatientCoverageDistrib(df,x = 'CombFeat_idx',xtick=50,saveFig=False):
    funcs = [min,np.median,max]
    m = df[x].max()
    for func in funcs:
        summarizeGroupCumPlot(df,func=func)
    plt.xticks(range(0,round(m,-2)+xtick,xtick))
    plt.yticks(np.linspace(0,1.0,11))    
    plt.xlabel('Peptide Rank Per Patient')    
    if saveFig:
        plt.savefig(saveFig.format('PatientCoverage'),dpi=600)
    plt.show()

def patientWiseEpitopeDistrib(df,x = 'CombFeat_idx',xtick=50,saveFig=False):
    m = df[x].max()
    sns.swarmplot(data=df,y='Patient', x = x,orient='h',color='Black')
    sns.boxplot(data=df,y='Patient', x = x,orient='h')
    plt.xticks(range(0,round(m,-2)+xtick,xtick))
    plt.xlabel('Peptide Rank Per Patient')    
    if saveFig:
        plt.savefig(saveFig.format('PatientBoxplot'),dpi=600)
    plt.show()

def plotCumulativeTargetRankDistrib(df,xtick=50,saveFig=False):
    cumulativeRankDistrib(df,xtick=xtick,saveFig=saveFig)
    cumulativePatientCoverageDistrib(df,xtick=xtick,saveFig=saveFig)
    patientWiseEpitopeDistrib(df,xtick=xtick,saveFig=saveFig)


def clusterAndGetPartitionCol(df,k=8,peptCol='PeptMut'):
    testIDXs,trainIDXs = partitionTools.commonMotifPartitionWrapper(df,k=k,peptCol=peptCol)
    df['Partition'] = partitionTools.getPartitionColumn(testIDXs)
    return df

def getTestTrain(df,part,partCol='Partition'):
    dfTest = df[df[partCol]==part].copy(deep=True)
    dfTrain = df[df[partCol]!=part].copy(deep=True)
    return dfTest,dfTrain


def getTestTrain_rescaledIDX(df,part,featureDict,rescale=True,partCol='Partition'):
    dfTest,dfTrain = getTestTrain(df,part,partCol=partCol)
    if rescale:
        dfTest = rankMultiFeaturePerPatient(dfTest,featureDict)
        dfTrain = rankMultiFeaturePerPatient(dfTrain,featureDict)
    return dfTest,dfTrain

def CVevalFeatureSelect(df,featureDict,k=5,partCol='Partition',aggFunc=scipy.stats.mstats.gmean,combNum=5,X=50,rescale=True,partition=True,returnFeatlist = False):
    #df = clusterAndGetPartitionCol(df)
    if partition:
	    testIDXs,trainIDXs,df = partitionTools.commonMotifPartitionWrapper(df,k=8,peptCol='PeptMut',addPartCol=True)    
    dfList = []
    featureList = []
    df = rankMultiFeaturePerPatient(df,featureDict)
    for part in range(k):
        print(part)
        dfTest,dfTrain = getTestTrain_rescaledIDX(df,part,featureDict,rescale=rescale,partCol=partCol)
        bestFeats_part = testFeatureCombinations_greedy(dfTrain,combNum,aggFunc=aggFunc,X=X)[0]
        print(bestFeats_part)
        bestFeats_part_list = list(bestFeats_part[0])
        featureList += [bestFeats_part]
        dfTest = genCombRankFeats(dfTest,bestFeats_part_list,aggFunc=aggFunc)
        dfList.append(dfTest)
    dfOut = pd.concat(dfList)
    dfOut = rankMultiFeaturePerPatient(dfOut,featureDict)
    dfOut = rankByFeaturePerPatient(dfOut,feature='CombFeat',ascending=True)
    if returnFeatlist:
    	return dfOut,featureList
    return dfOut


def getTopFreqFeature(List,i = 1):
    countDict = {}
    for feat in List:
        try:
            countDict[feat]+=1
        except KeyError:
            countDict[feat] = 1
    topFeatures,count = zip(*sorted(list(countDict.items()),key=lambda val: val[1],reverse=True)[:i])
    return topFeatures

def cvFeatureSelection_meanRank(df,featureDict,numIter=False):
    featureSelDict = {}
    yieldDicts_features = {}
    if not numIter:
        numIter = len(featureDict)+1
    for i in range(1,numIter):
        dfOut,featureList = CVevalFeatureSelect(df,featureDict,partition=False,combNum=i,rescale=True,returnFeatlist=True)
        feats,top50yields,patientHits = zip(*featureList)
        featureSelDict[i] = getTopFreqFeature(feats[0],i=i)
        yieldDicts_features[i] = yieldsWrapperStats(dfOut,hue=True,targCol='Target',plot=True)

    yieldDF_features = genYieldDF(yieldDicts_features,list(range(1,len(featureDict)+1)))
    return yieldDF_features,featureSelDict
