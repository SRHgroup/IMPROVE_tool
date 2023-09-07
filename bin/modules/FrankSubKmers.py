import pandas as pd
import numpy as np
import os
import NNanalysis
import matplotlib.pyplot as plt
import scipy.stats


def getOverlapPreds(i,predsDict,epiLen,procFunc,epiIDX,overlap=9,coreLen=9,ks=range(13,22),kmerCondition='overlap'):
    overlapVals = []
    kmerSetStats = []
    epiLenKmerSetStats = []
    for k,preds in predsDict.items():
        if k not in ks:
            continue
        startOverlap = i+overlap-k
        start = int(max(0,startOverlap))
        
        endOverlap = i+(epiLen-overlap)+1
        end = int(min(endOverlap,len(preds)))
        for j in range(start,end):
            if kmerCondition=='simple':
                if i==j and k==epiLen:
                    overlapVals.append(preds[j][0])
                    kmerSetStats.append([k,preds[j][0]])
                    if k==epiLen and i in epiIDX:
                        epiLenKmerSetStats.append([k,preds[i][0]])            
            if kmerCondition=='overlap':
                overlapVals.append(preds[j][0])
                kmerSetStats.append([k,preds[j][0]])
                if k==epiLen and i in epiIDX:
                    epiLenKmerSetStats.append([k,preds[i][0]])
            elif kmerCondition=='coreOverlap':
                offset = preds[j][1]
                if (j+offset)>=i and (j+offset+coreLen)<=(i+epiLen):
                    #If binding core is fully within the reference peptide, include in set
                    overlapVals.append(preds[j][0])
                    kmerSetStats.append([k,preds[j][0]])
                    if k==epiLen and i in epiIDX:
                        epiLenKmerSetStats.append([k,preds[i][0]])
            elif kmerCondition=='coreIdentical':
                    offset = preds[j][1]
                    refOffset = predsDict[epiLen][i][1]
                    if (j+offset)==(i+refOffset):
                        #If binding core is identical to reference core, include in set
                        overlapVals.append(preds[j][0])
                        kmerSetStats.append([k,preds[j][0]])
                        if k==epiLen and i in epiIDX:
                            epiLenKmerSetStats.append([k,preds[i][0]])
    if len(epiLenKmerSetStats)>0 and False:
        x,y=zip(*kmerSetStats)
        xn = 0.1*np.random.randn(len(x))
        x=x+xn
        plt.scatter(x,y)
        x,y=zip(*epiLenKmerSetStats)
        plt.scatter(x,y,c='r',s=100)
        plt.show()
    if len(overlapVals)==0:
        overlapVals=predsDict[epiLen][i][0]
    return procFunc(overlapVals)


def procOverlapPreds(epiLen,predsDict,epiIDX,procFunc=lambda x:np.median(x),ks=range(13,22),kmerCondition='overlap'):
    return [getOverlapPreds(i,predsDict,epiLen,procFunc,epiIDX,ks=ks,kmerCondition=kmerCondition) for i in range(len(predsDict[epiLen]))]

def getOverlapFrank_compare(procPred,epiIDX):
    if len(set(procPred))==1:#If all values are the same
        return 50.0
    if procPred[epiIDX]==0.0:
        return 50.0
    return np.mean([p>procPred[epiIDX] for p in procPred])*100

def getOverlapFrank_index(procPred,epiIDX):
    l = len(procPred)
    oneHot  = [1 if i==epiIDX else 0 for i in range(l)]
    zipped = list(zip(procPred,oneHot))
    zipped = sorted(zipped)
    procPred_sorted,oneHot_sorted = zip(*zipped)
    epiIDX_sorted = np.argmax(oneHot_sorted)
    return (1-(epiIDX_sorted/l))*100
    #return np.mean(procPred>procPred[epiIDX])*100

def epiPredPlot(preds,epiIDX,franks,show=True):
    epiPreds = [preds[i] for i in epiIDX]
    plt.plot(preds)
    plt.scatter(epiIDX,epiPreds,c='r',s=300,marker='X')
    plt.title(f'Franks: {list(map(lambda x: round(x,3),franks))}')
    if show:
        plt.show()

def getOverlapFranks(epiLen,epiIDX,predsDict,procFunc=lambda x:np.median(x),plot=False,show=True,ks=range(13,22),frankCompare=True,kmerCondition='overlap'):
    procPred = procOverlapPreds(epiLen,predsDict,epiIDX,procFunc=procFunc,ks=ks,kmerCondition=kmerCondition)
    if frankCompare:
        franks = [getOverlapFrank_compare(procPred,IDX) for IDX in epiIDX]
    else:
        franks = [getOverlapFrank_index(procPred,IDX) for IDX in epiIDX]
    if plot:
        epiPredPlot(procPred,epiIDX,franks,show=show)
    return procPred, list(zip(epiIDX,franks))

def getEpiInd(l):
    return [i for i,j in enumerate(l) if bool(int(j))]

def makePeptLenPredDict(df,lenCol='PeptLen',predCol='Prediction',offsetCol='Offset',peptCol='Peptide',ks=range(13,22)):
    predsDict = {}
    for peptLen,lenDF in df.groupby('PeptLen'):
        if peptLen not in ks and lenDF['Measure'].values.sum()>0:
            ks = list(ks)+[peptLen]
            #print(peptLen)
        elif peptLen not in ks:
            continue
        preds = lenDF[[predCol,offsetCol,peptCol]].values
        predsDict[peptLen] = dict(zip(range(len(preds)),preds))
    return predsDict,ks

def frankOverlapWrapper(df,procFunc = lambda x:np.median(x),plot=False,show=True,ks=range(13,22),frankCompare=True,kmerCondition='overlap',predCol='Prediction'):
    evalList = []
    #predsDict = dict(list(df.groupby('PeptLen')['Prediction']))
    predsDict,ks = makePeptLenPredDict(df,ks=ks,predCol=predCol)
    for plen,groupDF in df.groupby('PeptLen'):
        measure = groupDF['Measure'].values
        if sum(measure)>0:#If there are epitopes in this kmer set
            epiInds = getEpiInd(measure)
            epiSeqs = [groupDF['Peptide'].values[ind] for ind in epiInds]
            preds, franks = getOverlapFranks(int(plen),epiInds,
                                            predsDict,procFunc=procFunc,
                                            plot=plot,show=show,ks=ks,
                                            frankCompare=frankCompare,
                                            kmerCondition=kmerCondition)
            epiPreds = [preds[i] for i in epiInds]
            idx,franks = zip(*franks)
            evals = list(zip(epiSeqs,epiPreds,idx,franks))
            evalList.append(evals)
    evalList = [e for evals in evalList for e in evals]#Unpack list of lists to a list
    #print(evalList)
    return evalList

def procNetOutDF(df,rank=False):
    df['PeptLen'] = df['Peptide'].apply(len)
    df['Prediction'] = df['Prediction'].apply(float)
    df['Measure'] = df['Measure'].apply(float)
    df['Offset'] = df['Offset'].apply(int)
    if rank:
        df['Prediction'] = 100-df['Prediction']
    #return df[['Peptide','PeptLen','Prediction','Measure','Offset','MHC']]
    return df

def epiEvalFilenameProc(filename,splitter='__',simple=False):
    fileProc = os.path.splitext(filename)[0]
    fileProc = fileProc.split('---')[1]
    split = fileProc.split(splitter)
    if simple:
        return split[-3:-1]
    else:
        return split[-2:]

#####################################

def epiDir2FrankEvalDF(epiDir,model='nnalign',procFunc = lambda x:np.max(x),plot=False,show=True,simple=False,ks=range(13,22),frankCompare=True,kmerCondition='overlap',predCol='Prediction',rank=False):
    gatherList = []
    logFiles = [filename for filename in os.listdir(epiDir) if filename.startswith('log_eval')]
    evalTot = len(logFiles)
    evalCount = 0
    for evalCount, filename in enumerate(logFiles):    
        if evalCount%100==0:
            print(f'File {evalCount} out of {evalTot}')
        alid,uid = epiEvalFilenameProc(filename,simple=simple)
        df = NNanalysis.networkOut2DF(epiDir,filename,model=model)
        df = procNetOutDF(df,rank=rank)
        evals = frankOverlapWrapper(df,procFunc = procFunc,plot=plot,show=show,ks=ks,frankCompare=frankCompare,kmerCondition=kmerCondition,predCol=predCol)
        #print(evals)
        try:
            #for eval in evals[0]:
            for eval in evals:
                seq,pred,idx,frank = eval
                allele = df[df['Peptide']==seq]['MHC'].values[0]
                bc = df[df['Peptide']==seq]['Binding_core'].values[0]
                gatherList.append((alid,allele,uid,seq,bc,pred,idx,frank))
        except IndexError:
            print("Index error on eval for file: {}".format(filename))
            pass        
    dfOut = pd.DataFrame(gatherList,columns=['ALID','MHC','UID','Peptide','Binding_core','Prediction','IDX','Frank'])
    if rank:
        dfOut['Prediction'] = 100 - dfOut['Prediction']
    return dfOut


def labelCols(dfList,tagList,cols=['Version','Data','Context','Eval']):
    dfListProc = []
    for i,df in enumerate(dfList):
        for j,col in enumerate(cols):
            df[col] = tagList[i][j]
        dfListProc.append(df)
    return pd.concat(dfListProc)

def runMultiFunc(datDir,funcList,model='nnalign',plot=False,show=True,simple=False,ks=range(13,22),frankCompare=True,kmerCondition='overlap'):
    return [epiDir2FrankEvalDF(datDir,procFunc=func,model=model,plot=plot,show=show,simple=simple,ks=ks,frankCompare=frankCompare,kmerCondition=kmerCondition) for func in funcList]

def runMultiFunc_wrapper(dirList,funcList,tagList,cols=['Train','Eval'],model='nnalign',plot=False,show=True,modelList=False,simple=False,ks=range(13,22),frankCompare=True,kmerCondition='overlap'):
    dfList = []
    for i,datDir in enumerate(dirList):
        print("Processing {} out of {} directories".format(i+1,len(dirList)))
        if modelList:
            model = modelList[i]
        funcResList = runMultiFunc(datDir,funcList,model=model,plot=plot,show=show,simple=simple,ks=ks,frankCompare=frankCompare,kmerCondition=kmerCondition)
        dfList.extend(funcResList)
    return labelCols(dfList,tagList,cols=cols)

def getMutualFranks(df,valThresh=20,valCol='Frank',categoryCol = 'Method',inv=False,groupCols = ['Allele','UID','Peptide']):
    dfList = []
    for valTup, group_df in df.groupby(groupCols):
        numCats = len(set(df[categoryCol].values))
        if len(group_df)!=numCats:
            continue
        minVal = min(group_df[valCol].values)
        #print(minVal)
        if inv:
            if minVal > valThresh:
                dfList.append(group_df)
        else:
            if minVal < valThresh:
                dfList.append(group_df)
    return pd.concat(dfList)

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

def pivotFrankResults(df,pivotCol = 'Method',compareCol='Frank',uniqCol=['Allele','UID','Peptide']):
    df['Unique'] = ['--'.join([a,u,p]) for (a,u,p) in df[uniqCol].values]
    pivot_df  = df[['Unique',compareCol,pivotCol]].pivot(columns=pivotCol,index='Unique',values=compareCol)
    pivot_df = pivot_df.reset_index()
    uSplit = [u.split('--') for u in pivot_df['Unique'].values]
    pivot_df['Allele'],pivot_df['UID'],pivot_df['Peptide'] = zip(*uSplit)
    return pivot_df