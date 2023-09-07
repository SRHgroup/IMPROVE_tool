import os
import pandas as pd
import NNanalysis
import FrankAnalysis
import FrankSubKmers
import scipy

def dfClassLenFilter(df,col,count,gt=True):
    if not gt:
        return pd.concat([group[1] for group in df.groupby(col) if len(group[1])<count])
    return pd.concat([group[1] for group in df.groupby(col) if len(group[1])>count])

def dfGTLTfilter(df,col,thresh,gt=True):
    if gt:
        return df[df[col]>=thresh]
    else:
        return df[df[col]<thresh]

def dfLenFilter(df,col,thresh,gt=True):
    if gt:
        return df[df[col].apply(lambda x:len(x)>=thresh)]
    else:
        return df[df[col].apply(lambda x:len(x)<thresh)]

def dfColContainsFilter(df,cond,col,neg=False):
    if neg:
        return df[[not cond in x for x in df[col].values]]
    return df[[cond in x for x in df[col].values]]

def dfColContainsAnyFilter(df,condList,col,neg=False):
    if neg:
        return df[df[col].apply(lambda x: not any([y in x for y in condList]))]
    else:
        return df[df[col].apply(lambda x: any([y in x for y in condList]))]

def dfColMatchAnyFilter(df,condList,col,neg=False):
    if neg:
        return df[df[col].apply(lambda x: not any([y == x for y in condList]))]
    else:
        return df[df[col].apply(lambda x: any([y == x for y in condList]))]

def dfColRemoveOverlap(df,condList,col,neg=False):
    if neg:
        return df[df[col].apply(lambda x: x in condList)]
    else:
        return df[df[col].apply(lambda x: not x in condList)]

def dfColFilter(df,cond,col,neg=False):
    if neg:
        return df[df[col]!=cond]
    return df[df[col]==cond]

def groupCount(df,groupCol='AlleleList_ID',rename='Count'):
    return df.groupby(groupCol).apply(len).reset_index().rename(columns={0:rename}).sort_values(rename,ascending=False)

def colCounter(df,col):
	"""Count instances of all values in a selected columns of a df"""
	countDict = {}
	for val in df[col].values:
		countDict[val] = countDict.get(val,0)+1
	return countDict

def getBigSets(df,col='AlleleList_ID',thresh=100):
    return pd.concat([dfGroup for groupName,dfGroup in df.groupby(col) if len(dfGroup)>thresh])

def labelCols(dfList,tagList,cols=['Version','Data','Context','Eval']):
    dfListProc = []
    for i,df in enumerate(dfList):
        for j,col in enumerate(cols):
            df[col] = [tagList[i][j]]*len(df)
        dfListProc.append(df)
    return pd.concat(dfListProc)

def summaryDFlistGet(datDir, evalDirList, datType='EL',splitter=['AlleleList_ID'],noSource=False,XAL=False,model='nnalign',predCol='Prediction'):
    summaryDFList = []
    for evalDir in evalDirList:
        summaryDFList.append(NNanalysis.cvDir2summaryDF(datDir,evalDir,
                                                        datType=datType,splitter=splitter,
                                                        noSource=noSource,XAL=XAL,model=model,predCol=predCol))
    return summaryDFList

def labelSummaryDFWrap(datDir,evalDirList,tagList,datType='EL',splitter=['AlleleList_ID'],
                        noSource=False,XAL=False,model='nnalign',predCol='Prediction'):
    summaryDFlist = summaryDFlistGet(datDir,evalDirList,
                                     datType=datType,splitter=splitter,
                                    noSource=noSource,XAL=XAL,model=model,predCol=predCol)
    summaryDFList =  labelCols(summaryDFlist,tagList)
    #if len(splitter)==1:
    #    summaryDFList = summaryDFList.reset_index().rename(columns={"index":splitter[0]})
    #else:
    #    summaryDFList = summaryDFList.reset_index().rename(columns={"level_{}".format(i):s for i,s in enumerate(splitter)})
    return summaryDFList

def summaryDFlistGet_kmers(datDir, evalDirList,emptyFiles,model='nnalign',splitter='__',predCol='Prediction'):
    summaryDFList = []
    for evalDir in evalDirList:
        summaryDFList.append(FrankAnalysis.kmerDir2epiStatsDF(os.path.join(datDir,evalDir),emptyFiles,model=model,splitter=splitter,predCol=predCol))
    return summaryDFList

def labelSummaryDFWrap_kmers(datDir,evalDirList,tagList,model='nnalign',splitter='__',predCol='Prediction',rank=False):
    emptyFiles = FrankAnalysis.getEmptyLogs(datDir)
    summaryDFlist = summaryDFlistGet_kmers(datDir,evalDirList,emptyFiles,model=model,splitter=splitter,predCol=predCol)
    summaryDFlist_lab =  labelCols(summaryDFlist,tagList)
    if rank:
        summaryDFlist_lab['Frank'] = summaryDFlist_lab['Frank'].apply(lambda x: 100-x)
    return summaryDFlist_lab

def summaryDFlistGet_kmerSub(datDir,evalDirList,**kmerSubKW):
    summaryDFList = []
    for evalDir in evalDirList:        
        summaryDFList.append(FrankSubKmers.epiDir2FrankEvalDF(os.path.join(datDir,evalDir),**kmerSubKW))
    return summaryDFList

def labelSummaryDFWrap_kmerSub(datDir,evalDirList,tagList,**kmerSubKW):
    summaryDFlist = summaryDFlistGet_kmerSub(datDir,evalDirList,**kmerSubKW)
    summaryDFlist_lab =  labelCols(summaryDFlist,tagList)
    return summaryDFlist_lab

def procNNaddXALenFilter(df,XAL=True,thresh=100):
    if XAL:
        df['Data'] = df['Alellelist_ID'].apply(lambda x:'SA-Data' if len(x.split('__'))<3 else 'MA-Data' )
    df = df[df['PosLig#'].apply(lambda x:x>thresh)]
    return df

def labelSummaryDFWrap_thresh(datDir, evalDirList,tagList,splitter=['AlleleList_ID'],model='nnalign',thresh=100,XAL=True):
    summaryDFList = labelSummaryDFWrap(datDir,evalDirList,tagList,model=model,splitter=splitter)
    return procNNaddXALenFilter(summaryDFList,XAL=XAL,thresh=thresh)

def colBinomTest(df,Col1,Col2,rounder=3,inv=False,alternative='two-sided'):
    dfDelta = (df[Col1]-df[Col2]).apply(lambda x:round(x,rounder))
    if inv:
        dfDelta = -dfDelta    
    nonZeroDelta = [bool(delta) for delta in dfDelta.values]
    N = len(dfDelta)
    ties = N-sum(nonZeroDelta)
    dfDelta_excludeTies = dfDelta[nonZeroDelta]
    n = len(dfDelta_excludeTies)
    wins = sum([delta > 0 for delta in dfDelta_excludeTies.values])
    binom_results = scipy.stats.binom_test(wins,n,alternative=alternative)
    print("N:{}, Ties:{}, n:{}, wins:{}".format(N,ties,n,wins))
    return binom_results

def calcBinom(summaryDFList,compCol='AUC',pivotCol='Eval',rounder=3,win='Sum-p10',lose='Max',alternative='two-sided'):
    AUCcomp = summaryDFList[[pivotCol,'ALID',compCol]].pivot(index='ALID',columns=pivotCol,values=compCol)
    print(colBinomTest(AUCcomp,win,lose,alternative=alternative,rounder=rounder))
    return AUCcomp

def groupedColBinomTest(df,class1,class2,groupCol='DQ',pivotCol='Data',compCol='PPV'):
    for g,dfG in df.groupby(groupCol):
        dfG_pivot = pivotSummaryDF(dfG,compCol=compCol,pivotCol=pivotCol)
        print(g)
        print(colBinomTest(dfG_pivot,class1,class2))

def groupCountMergeCompare(df1,df2,groupCol='PeptLen',renameCol='Count'):
    df1 = df1.groupby(groupCol).apply(len).reset_index()    
    df2 = df2.groupby(groupCol).apply(len).reset_index()
    name1,name2="{}_1".format(renameCol),"{}_2".format(renameCol)
    dfMerge = df1.merge(df2,on=groupCol,how='outer').fillna(0).rename(columns={'0_x':name1,'0_y':name2})
    dfMerge['Ratio'] = dfMerge[name2]/dfMerge[name1]
    return dfMerge

def pivotSummaryDF(summaryDFList,indexCol='AlleleList_ID',compCol='AUC',pivotCol='Eval'):
    return summaryDFList[[pivotCol,indexCol,compCol]].pivot(index=indexCol,columns=pivotCol,values=compCol)

def concatLabelDFs(dfList,labels,labelCol):
    dfListOut = []
    for i,df in enumerate(dfList):
        dfOut = df.copy(deep=True)
        dfOut[labelCol] = labels[i]
        dfListOut.append(dfOut)
    return pd.concat(dfListOut)

def concatPivotDFs(dfList,labels,labelCol='Data',indexCol='MHC',compCol='PPV'):
    concatDF = concatLabelDFs(dfList,labels,labelCol)
    return pivotSummaryDF(concatDF,indexCol=indexCol,compCol=compCol,pivotCol=labelCol).reset_index()[[indexCol,*labels]]#.drop('index',axis=1)