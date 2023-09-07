### Generic Modules
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import datetime
import multiprocessing

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

### In house modules
import rankEpiTools

def timeit(method):
    def timed(*args, **kw):
        ts = datetime.datetime.now()
        result = method(*args, **kw)
        te = datetime.datetime.now()
        timeDelta = te-ts
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = timeDelta
        else:
            print('{} {}'.format(method.__name__, timeDelta))
        return result
    return timed

def getSKlearnModel(modelType,**modelKW):
    if modelType=='RF':
        return RandomForestClassifier(**modelKW)
    elif modelType=='LogR':
        return LogisticRegression(**modelKW)

    elif modelType=='NN':
        return MLPClassifier(**modelKW)
    elif modelType=='GB':
        return GradientBoostingClassifier(**modelKW)
    elif modelType=='XGB':
        return XGBClassifier(use_label_encoder=False,eval_metric='logloss',**modelKW)
    elif modelType=='XGBRF':
        return XGBRFClassifier(use_label_encoder=False,eval_metric='logloss',**modelKW)
    elif modelType=='KNN':
        if 'modelArgs' in modelKW.keys():
            modelArgs = modelKW.pop('modelArgs')
            return KNeighborsClassifier(*modelArgs,**modelKW)
        else:
            return KNeighborsClassifier(**modelKW)
    elif modelType=='SVC':
        return SVC(**modelKW)
    elif modelType=='Ada':
        return AdaBoostClassifier(**modelKW)
    elif modelType=='QDA':
        return QuadraticDiscriminantAnalysis(**modelKW)    

def trainSKlearnModel(model,dfTrain,features,targCol):
    X_train = dfTrain[features].values
    Y_train = dfTrain[targCol].apply(int).values
    model.fit(X_train,Y_train)
    return model

def evalSKlearnModel(dfEval,features,model,outCol='classPred'):
    dfOut = dfEval.copy(deep=True)
    try:
        dfOut[outCol] = model.predict_proba(dfOut[features].values)[:,1]
    except AttributeError:
        dfOut[outCol] = model.predict(dfOut[features].values)        
    return dfOut

def singlePartitionTraining(df,fold,features,featureDict,targCol='Target',modelType='NN',outCol='classPred',AB=False,**modelKW):
    dfTest, dfTrain = rankEpiTools.getTestTrain_rescaledIDX(df,fold,featureDict,rescale=False)
    if AB:
        dfTrain = dfTrain[dfTrain['Loci']!='C'].copy(deep=True)    
    model = getSKlearnModel(modelType,**modelKW)
    model = trainSKlearnModel(model,dfTrain,features,targCol)
    dfOut = evalSKlearnModel(dfTest,features,model,outCol=outCol)
    return (dfOut,model)

def runCrossValidationTraining_multiProcess(df,features,featureDict,targCol='Target',folds=5,outCol='classPred',modelType='NN',partCol='Partition',**modelKW):
    
    folds = df[partCol].unique()
    print(folds)
    numProcs = multiprocessing.cpu_count()-1
    with multiprocessing.Pool(processes=numProcs) as pool:
        dfOutModel_list = [pool.apply(singlePartitionTraining,(df,fold,features,featureDict),{'modelType':modelType,'targCol':targCol,'outCol':outCol,**modelKW}) for fold in folds]

    dfList,modelList = zip(*dfOutModel_list)
    modelDict = {'models':modelList}
    return pd.concat(dfList),modelDict

def runCrossValidationTraining(df,features,featureDict=False,targCol='Target',folds=5,outCol='classPred',modelType='NN',**modelKW):
    modelDict = {'models':[]}
    dfList = []
    for i in range(folds):
        if not featureDict:
            dfTest,dfTrain = rankEpiTools.getTestTrain(df,i)
        else:
            dfTest, dfTrain = rankEpiTools.getTestTrain_rescaledIDX(df,i,featureDict,rescale=False)
        if len(dfTest)==0:#Skip empty fold in inner loop of nested cross validation
            continue
        model = getSKlearnModel(modelType,**modelKW)
        model = trainSKlearnModel(model,dfTrain,features,targCol)
        dfOut = evalSKlearnModel(dfTest,features,model,outCol=outCol)
        
        dfList.append(dfOut)
        modelDict['models'].append(model)
    return pd.concat(dfList),modelDict

def meanCVeval(df,features,modelDict):
    dfList = []
    for i,model in enumerate(modelDict['models']):
        dfPred = evalSKlearnModel(df,features,model)
        dfPred['CVi'] = i
        dfList.append(dfPred)
    dfConcat = pd.concat(dfList)

    meanPredDF = dfConcat.groupby(['Patient','HLA','PeptMut','Target']).apply(lambda dfG: dfG['classPred'].mean()).reset_index().rename(columns={0:'meanPred'})
    return df.merge(meanPredDF)


def computeROCAUC(df,targCol='Target',predCol='classPred'):
    return sklearn.metrics.roc_auc_score(df[targCol], df[predCol])

def getPPV_pos(y,pred,posNum=False):
    if not posNum:
        posNum = sum([int(i) for i in y]) 
    zipPred = list(zip(pred,y))
    zipPred.sort(reverse=True)
    pred_top,y_top = list(zip(*zipPred[:posNum]))
    top_pos = sum([int(i) for i in y_top])
    PPV = float(top_pos)/posNum
    return round(PPV,3)

def getPPV(df,predCol='classPred',targCol='Target',ascending=True):
    numPos = df[targCol].sum()
    if ascending:
        ppv = df.nlargest(int(numPos),predCol)[targCol].sum()/numPos
    else:
        ppv = df.nsmallest(int(numPos),predCol)[targCol].sum()/numPos
    return round(ppv,3)

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
            #dfOut,modelDict = runCrossValidationTraining_multiProcess(df,features_current,featureDict,modelType=modelType,**modelKW)            
            yieldDict = rankEpiTools.yieldsWrapperStats(dfOut,hue=True,targCol='Target',predCol='classPred',plot=False,rank=False,Print=False)
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
    return pd.DataFrame(bestFeatureScore_dfList).sort_values(sortCol,ascending=False)

def yieldHyperParamCombs(featureDict):
    """Yield KW dicts with all combinations of model hyperparameters"""
    allNames = sorted(featureDict)
    combinations = itertools.product(*(featureDict[Name] for Name in allNames))
    sortedFeats = sorted(list(featureDict.keys()))    
    for comb in combinations:    
        yield {sortedFeats[i]:feat for i,feat in enumerate(comb)}

def getTopFeatures(df,paramDict=False, sortCol=['Epitope-Hits','Patient-Hits'],featureCol='Features_current'):
    df = df.sort_values(sortCol,ascending=False)
    features_opt = df.iloc[0,:][featureCol]
    if paramDict:
        paramDict_opt = {key:df.iloc[0,:][key] for key in paramDict.keys()}
        return df,features_opt,paramDict_opt
    else:
        return df,features_opt

        
def hyperParamModelSelectCV(df,featureDict,paramDict,maxFeat=6,sortCol = ['Epitope-Hits','Patient-Hits'],modelType='NN'):
    dfList = []    
    for modelKW in yieldHyperParamCombs(paramDict):
        print(modelKW)
        dfFF = forwardFeatureSelectionCV(df,featureDict,maxFeat=maxFeat,sortCol=sortCol,modelType=modelType,**modelKW)
        for param,value in modelKW.items():
            dfFF[param] = value
        dfList.append(dfFF)
    dfOut = pd.concat(dfList)
    dfOut,features_opt,paramDict_opt = getTopFeatures(dfOut,paramDict,sortCol=sortCol,featureCol='Features_current')
    return dfOut,features_opt,paramDict_opt


def runTrainCVeval(dfTrain,dfEval,featuresOpt,featureDict,modelType='NN',**paramDict):
    dfTrain_cvEval,modelDict = runCrossValidationTraining(dfTrain,featuresOpt,featureDict,modelType=modelType,**paramDict)
    dfTrain_cvEval['meanPred'] = dfTrain_cvEval['classPred']
    datasetTrain = '__'.join(dfTrain['Dataset'].unique())
    saveFig = os.path.join(figDir,'yieldPlots_{}_ABC_{}.png'.format(datasetTrain,modelType))
    yieldDict_train = rankEpiTools.yieldsWrapperStats(dfTrain_cvEval,hue=True,targCol='Target',predCol='meanPred',plot=True,rank=False)
    yieldDict_train['Dataset'] = datasetTrain
    yieldDict_train['Model'] = modelType

    dfEval_cvEval = meanCVeval(dfEval,featuresOpt,modelDict)

    yieldDictList = [yieldDict_train]
    for dataset,dfG in dfEval_cvEval.groupby('Dataset'):
        saveFig = os.path.join(figDir,'yieldPlots_{}_{}.png'.format(dataset,modelType))
        X = 20 if dataset=='Tesla' else 50
        yieldDict = rankEpiTools.yieldsWrapperStats(dfG,hue=True,targCol='Target',predCol='meanPred',plot=True,rank=False,saveFig=saveFig,X=X)
        yieldDict['Dataset'] = dataset
        yieldDict['Model'] = modelType        
        yieldDictList.append(yieldDict)
    yieldDF = pd.DataFrame(yieldDictList)
    dfCVeval = pd.concat([dfTrain_cvEval,dfEval_cvEval])
    dfCVeval['Model'] = modelType
    return dfCVeval, yieldDF

def runTrainCVeval_meanRank(dfEval,featuresOpt,featureDict,modelType='MeanRank',plot=True):
    dfList = []
    yieldDictList = []
    for dataset,dfG in dfEval.groupby('Dataset'):
        X = 20 if dataset=='Tesla' else 50
        dfEval_meanRank = rankEpiTools.evalDFonMultiFeatureWrapper(dfG,featuresOpt,featureDict)
        dfList.append(dfEval_meanRank)
        yieldDict = rankEpiTools.yieldsWrapperStats(dfEval_meanRank,hue=True,targCol='Target',plot=plot,show=True,saveFig=False,X=X)
        yieldDict['Dataset'] = dataset
        yieldDict['Model'] = modelType        
        yieldDictList.append(yieldDict)
    yieldDF = pd.DataFrame(yieldDictList)
    dfCVeval = pd.concat(dfList)
    dfCVeval['Model'] = modelType
    return dfCVeval, yieldDF

def getRandomSampleYieldDict(df,sampling=10,targCol='Target',predCol='randPred',X=50):
    yieldDictList = []
    for dataset,dfG in df.groupby('Dataset'):
        for i in range(sampling):
            dfG = addRandCol(dfG)
            X= 20 if dataset=='Tesla' else 50
            yieldDict_random = rankEpiTools.yieldsWrapperStats(dfG,hue=True,targCol=targCol,predCol=predCol,plot=False,show=False,saveFig=False,Print=False,X=X)
            yieldDict_random['Dataset'] = dataset
            yieldDictList.append(yieldDict_random)
    yieldDF = pd.DataFrame(yieldDictList).groupby('Dataset').apply(lambda dfG: dfG.mean(axis=0)).reset_index()
    yieldDF['Model'] = 'Random'
    return yieldDF

def addRandCol(df,randCol='randPred',model='Rand'):
    df[randCol] = np.random.random(len(df))
    df['Model'] = model
    return df.copy(deep=True)

def concatYieldDF_hueBarplot(df,saveFig=False):
    plt.figure(figsize=(18,8))
    sns.barplot(data=df,x='Patient',y='value',hue='Model')
    plt.ylabel("Epitopes")
    plt.xticks(rotation=30)
    plt.yticks(range(0,int(df['value'].max()),2))
    if saveFig:
        plt.savefig(saveFig,dpi=600)
    plt.show()

def getModelMergedYieldDict2(dfList_in, plot = True,saveFig=False):
    dfList = []
    for modelDict in dfList_in:
        dfYields = rankEpiTools.getFeatureYields(modelDict['df'],predCol=modelDict['predCol'],rank=modelDict['rank'],X=modelDict['X'])
        dfYields['Model'] = modelDict['Model']
        dfList.append(dfYields)
    concatYieldDF = pd.concat(dfList)
    
    concatYieldDF_allEpi = concatYieldDF[['Patient','Target','Model']].melt(id_vars=['Patient','Model']).drop_duplicates('Patient')
    concatYieldDF_allEpi['Model'] = 'Total'
    DFmelt = concatYieldDF[['Patient','Target_top50','Model']].melt(id_vars=['Patient','Model'])
    concatYieldDF_all = pd.concat([concatYieldDF_allEpi,DFmelt])
    if plot:
        concatYieldDF_hueBarplot(concatYieldDF_all,saveFig=saveFig)
    return concatYieldDF_all

def applyGroupConcatYield(concatYieldDF_master,groupCol='Dataset'):
    dfList = []
    for group,concatYieldDF in concatYieldDF_master.groupby('Dataset'):

        concatYieldDF_allEpi = concatYieldDF[['Patient','Target','Model']].melt(id_vars=['Patient','Model']).drop_duplicates('Patient')
        concatYieldDF_allEpi['Model'] = 'Total'
        DFmelt = concatYieldDF[['Patient','Target_top50','Model']].melt(id_vars=['Patient','Model'])
        concatYieldDF_all = pd.concat([concatYieldDF_allEpi,DFmelt])
        concatYieldDF_all[groupCol] = group
        
        if plot:
            concatYieldDF_hueBarplot(concatYieldDF_all,saveFig=saveFig)
        dfList.append(concatYieldDF_all)
    return pd.concat(dfList)

def getModelMergedYieldDict(dfMaster,modelDicts, plot = True,saveFig=False):
    dfList = []
    for dataset, dfD in dfMaster.groupby('Dataset'):
        for model,dfG in dfD.groupby('Model'):
            #print(model)
            modelDict = modelDicts[model]
            X = 20 if dataset=='Tesla' else 50
            dfYields = rankEpiTools.getFeatureYields(dfG,predCol=modelDict['predCol'],rank=modelDict['rank'],X=X)
            dfYields['Model'] = model
            dfYields['Dataset'] = dataset
            dfList.append(dfYields)
            
    concatYieldDF = pd.concat(dfList)
    concatYieldDF_out = applyGroupConcatYield(concatYieldDF,groupCol='Dataset')
    return concatYieldDF_out


def getPatientTypingFromDF(df,coveredAlleles):
    patientTyping = df.groupby('Patient').apply(lambda dfG: dfG['HLA'].unique()).reset_index().rename(columns={0:'Alleles'})
    patientTyping['HLA-Hits'] = patientTyping['Alleles'].apply(lambda alleles: sum([allele in coveredAlleles for allele in alleles]))
    patientTyping['Dataset'] = patientTyping['Patient'].apply(lambda pat: pat.split('-')[0])
    sns.countplot(data=patientTyping,x='HLA-Hits')
    plt.show()
    return patientTyping


def updatePatientYields(dfEval,dfYields):
    dfList = []
    for i in range(3):
        dfOut = dfEval[dfEval['HLA-Hits']>=i].copy(deep=True)
        patientsHLA = dfOut.groupby('Dataset').apply(lambda dfG: dfG['Patient'].nunique()).reset_index().rename(columns={0:'Patient-HLA'})
        patientsHLA['HLA-Hits'] = i
        dfList.append(patientsHLA)
    patientsHLA = pd.concat(dfList)
    
    patientsAll = dfEval.groupby('Dataset').apply(lambda dfG: dfG['Patient'].nunique()).reset_index().rename(columns={0:'Patient-All'})
    
    dfYields = dfYields.merge(patientsHLA)
    dfYields = dfYields.merge(patientsAll)
    dfYields['Patient Inclusion%'] = dfYields['Patient-HLA']/dfYields['Patient-All']*100
    dfYields['Patient Inclusion%'] = dfYields['Patient Inclusion%'].apply(lambda hit:round(hit,1))
    dfYields['Patient Hit%'] = dfYields['Patient-Hits']/dfYields['Patient-HLA']*100
    dfYields['Patient Hit%'] = dfYields['Patient Hit%'].apply(lambda hit:round(hit,1))    
    return dfYields

def groupYieldHLAHits(df,model='RF',hlaHitsRange=range(3)):
    yieldDictList = []
    for hlaHits in hlaHitsRange:
        for dataset,dfG in df.groupby('Dataset'):
            dfOut = dfG.copy(deep=True)
            dfOut = dfOut[dfOut['HLA-Hits']>=hlaHits]
            X = 20 if dataset=='Tesla' else 50
            yieldDict = rankEpiTools.yieldsWrapperStats(dfOut,hue=True,targCol='Target',predCol='meanPred',plot=False,rank=False,Print=False,X=X)
            yieldDict['Dataset']  = dataset
            yieldDict['HLA-Hits']  = hlaHits
            yieldDict['Model']  = model
            yieldDict['Model+HLA']  = "{}- HLA:{}".format(model,hlaHits)
            
            yieldDict_random = getRandomSampleYieldDict(dfOut).reset_index(drop=True)
            yieldDict_random = dict(list(zip(yieldDict_random.columns.values,yieldDict_random.values[0])))
            yieldDict_random['Dataset']  = dataset
            yieldDict_random['HLA-Hits']  = hlaHits
            yieldDict_random['Model']  = 'Random'
            yieldDict_random['Model+HLA']  = "{}- HLA:{}".format("Random",hlaHits)            
            
            yieldDictList.append(yieldDict)
            yieldDictList.append(yieldDict_random)            
    return pd.DataFrame(yieldDictList)


def hlaHitYieldsWrapper(dfEval,coveredAlleles,hlaHitsRange=range(3)):
    patientTyping = getPatientTypingFromDF(dfEval,coveredAlleles)
    dfEval = dfEval.merge(patientTyping[['Patient','HLA-Hits']]).copy(deep=True)
    dfEval_cov = dfTools.dfColContainsAnyFilter(dfEval,coveredAlleles,'HLA')
    #print(dfEval_cov.head())
    try:
        dfYields = groupYieldHLAHits(dfEval_cov,hlaHitsRange=hlaHitsRange)
    except KeyError:
        print(len(dfEval_cov))
        sys.exit()
    dfYields = updatePatientYields(dfEval,dfYields)
    return dfYields

        
def sumYieldsAcrossDatasets(dfYields):
    yieldDictList = []
    for (hlaHit,model), dfG in dfYields.groupby(['HLA-Hits','Model']):
        yieldSumDict = dict(dfG[['Epitope-Total','Epitope-Hits','Patient-Resp','Patient-Hits','Patient-Total','Patient-HLA','Patient-All']].sum(axis=0))
        yieldSumDict['HLA-Hits'] = hlaHit
        yieldSumDict['Model'] = model
        yieldDictList.append(yieldSumDict)
    dfYields = pd.DataFrame(yieldDictList)

    dfYields['Epitope Yield%'] = dfYields['Epitope-Hits']/dfYields['Epitope-Total']*100
    dfYields['Epitope Yield%'] = dfYields['Epitope Yield%'].apply(lambda hit:round(hit,1))    
    dfYields['Patient Inclusion%'] = dfYields['Patient-HLA']/dfYields['Patient-All']*100
    dfYields['Patient Inclusion%'] = dfYields['Patient Inclusion%'].apply(lambda hit:round(hit,1))
    dfYields['Patient Hit%'] = dfYields['Patient-Hits']/dfYields['Patient-HLA']*100
    dfYields['Patient Hit%'] = dfYields['Patient Hit%'].apply(lambda hit:round(hit,1))    
    dfYields['MeanEpiHits'] = dfYields['Epitope-Hits']/dfYields['Patient-Hits']
    dfYields['MeanEpiHits'] = dfYields['MeanEpiHits'].apply(lambda hit:round(hit,2))    

    return dfYields

def sumYieldsWrapper(df,coveredAlleles,hlaHitsRange=[0,1,2]):
    dfYields = hlaHitYieldsWrapper(df,coveredAlleles,hlaHitsRange=hlaHitsRange)
    return sumYieldsAcrossDatasets(dfYields)


def addNewAllelesYieldDict(df,coveredAlleles,newAlleles,hlaHitsRange=[0,1,2],addHLAnum=False):
    if not addHLAnum:
        addHLAnum = len(newAlleles)
    dfYields_sum = sumYieldsWrapper(df,coveredAlleles,hlaHitsRange=hlaHitsRange)
    dfYields_sum['Total-HLA'] = len(coveredAlleles)
    dfYields_sum['New-HLA'] = ''
    dfList = [dfYields_sum]
    #print(newAlleles[:addHLAs])
    for allele in newAlleles[:addHLAnum]:
        dfOut = df.copy(deep=True)
        coveredAlleles.append(allele)
        print(coveredAlleles)
        dfYields_sum = sumYieldsWrapper(dfOut,coveredAlleles,hlaHitsRange=hlaHitsRange)
        dfYields_sum['Total-HLA'] = len(coveredAlleles)
        dfYields_sum['New-HLA'] = allele
        dfList.append(dfYields_sum)
    dfYields_out = pd.concat(dfList)
    dfYields_out['Model-HLA'] = dfYields_out['Model'] + '-' + dfYields_out['HLA-Hits'].apply(str)
    return dfYields_out

def applyMeanPred(df,groupCols=['Patient','HLA','PeptMut','Target'],predCol='classPred',outCol='meanPred',multiCol='Seed'):
    df = df.copy(deep=True)
    meanPred = df.groupby(groupCols).apply(lambda dfG: dfG[predCol].mean()).reset_index().rename(columns={0:outCol})
    df = df.merge(meanPred)
    dfMean = df.drop_duplicates(groupCols)
    #dfMean = df[df[multiCol]==0]
    return dfMean

def yieldDictPerDataset(df,predCol='classPred',targCol='Target',groupCol='Dataset',plot=True):
    yieldDictList = []
    for dataset, dfG in df.groupby(groupCol):
        X = 20 if dataset=='Tesla' else 50
        yieldDict = rankEpiTools.yieldsWrapperStats(dfG,hue=True,targCol=targCol,X=X,predCol=predCol,plot=plot,rank=False)
        yieldDict['Dataset'] = dataset
        yieldDictList.append(yieldDict)
    return pd.DataFrame(yieldDictList)

def runSingleNestedCVpartition(df,fold,featureDict,maxFeat=5,sortCol = ['Epitope-Hits','Patient-Hits'],modelType='NN',targCol='Target',outCol='classPred',AB=False,modelKW={}):
    #Inner Loops: feature selection
    dfTest, dfTrain = rankEpiTools.getTestTrain_rescaledIDX(df,fold,featureDict,rescale=False)
    if AB:
        dfTrain = dfTrain[dfTrain['Loci']!='C'].copy(deep=True)
    yieldDF_features = forwardFeatureSelectionCV(dfTrain,featureDict,maxFeat=maxFeat,sortCol=sortCol,modelType=modelType,**modelKW)
    yieldDF_features['Fold'] = fold
    yieldDF_features,featuresOpt = getTopFeatures(yieldDF_features,paramDict=False,sortCol=sortCol,featureCol='Features_current')
    print(fold,featuresOpt)
    #Outer Loop: evaluation with selected model
    dfTest,model = singlePartitionTraining(df,fold,featuresOpt,featureDict,targCol=targCol,modelType=modelType,outCol=outCol,AB=AB,**modelKW)
    return (dfTest,model,featuresOpt)

@timeit
def NestedCVForwardFeatSelect_multiProcess(df,featureDict,maxFeat=5,sortCol = ['Epitope-Hits','Patient-Hits'],modelType='NN',targCol='Target',outCol='classPred',AB=False,plot=True,modelKW={}):
    folds = sorted(df['Partition'].unique())
    numProcs = multiprocessing.cpu_count()-1

    with multiprocessing.Pool(processes=numProcs) as pool:
        #dfTestModelFeatures_list = [pool.apply_async(runSingleNestedCVpartition,(df,fold,featureDict),{'maxFeat':maxFeat,'sortCol':sortCol,'modelType':modelType,'targCol':targCol,'outCol':outCol,'AB':AB,'modelKW':modelKW}) for fold in folds]
        results = [pool.apply_async(runSingleNestedCVpartition,(df,fold,featureDict),{'maxFeat':maxFeat,'sortCol':sortCol,'modelType':modelType,'targCol':targCol,'outCol':outCol,'AB':AB,'modelKW':modelKW}) for fold in folds]
        dfTestModelFeatures_list = [p.get() for p in results]
    dfList_CVeval,modelList,featuresOptList = zip(*dfTestModelFeatures_list)

    modelDict = {}
    for fold in folds:
        modelDict[fold] = {'Model':modelList[fold],'Features':featuresOptList[fold]}
    dfCVEval = pd.concat(dfList_CVeval)
    dfCVEval['Model'] = "{}{}".format(modelType,'-AB' if AB else '')
    yieldDF = yieldDictPerDataset(dfCVEval,plot=plot)
    yieldDF['Model'] = "{}{}".format(modelType,'-AB' if AB else '')
    return dfCVEval, yieldDF, modelDict

@timeit
def NestedCVForwardFeatSelect(df,featureDict,maxFeat=5,sortCol = ['Epitope-Hits','Patient-Hits'],folds=5,modelType='NN',targCol='Target',outCol='classPred',AB=False,plot=True,modelKW={}):
    dfList_CVeval = []
    dfList_yieldDFs = []
    modelDict = {}
    for fold in range(folds):
        #Inner Loops: feature selection
        dfTest, dfTrain = rankEpiTools.getTestTrain_rescaledIDX(df,fold,featureDict,rescale=False)
        if AB:
            dfTrain = dfTrain[dfTrain['Loci']!='C'].copy(deep=True)
        yieldDF_features = forwardFeatureSelectionCV(dfTrain,featureDict,maxFeat=maxFeat,sortCol=sortCol,modelType=modelType,**modelKW)
        yieldDF_features['Fold'] = fold
        yieldDF_features,featuresOpt = getTopFeatures(yieldDF_features,paramDict=False,sortCol=sortCol,featureCol='Features_current')
        print(fold,featuresOpt)
        
        #Outer Loop: evaluation with selected model
        model = getSKlearnModel(modelType,**modelKW)
        model = trainSKlearnModel(model,dfTrain,featuresOpt,targCol)
        dfTest = evalSKlearnModel(dfTest,featuresOpt,model,outCol=outCol)
        dfList_CVeval.append(dfTest)
        modelDict[fold] = {'Model':model,'Features':featuresOpt}

    dfCVEval = pd.concat(dfList_CVeval)
    dfCVEval['Model'] = "{}{}".format(modelType,'-AB' if AB else '')
    yieldDF = yieldDictPerDataset(dfCVEval,plot=plot)
    yieldDF['Model'] = "{}{}".format(modelType,'-AB' if AB else '')
    return dfCVEval, yieldDF, modelDict

def procMultiSeedDFEVal(dfEvalYieldModelList,numSeeds=5,modelType='RF',AB=False,noC=False,plot=True):
    dfCVEval_seed = pd.concat([EvalYieldModel[1] for EvalYieldModel in dfEvalYieldModelList])
    dfCVEval_seed = applyMeanPred(dfCVEval_seed)
    modelType = "{}{}".format(modelType,'-AB' if AB else '')
    dfCVEval_seed['Model'] = "{}-{}Seed".format(modelType,numSeeds)
    if noC:
        dfCVEval_seed = dfCVEval_seed[dfCVEval_seed['Loci']!='C']
    yieldDF = yieldDictPerDataset(dfCVEval_seed,predCol='meanPred',plot=plot)
    yieldDF['Model'] = "{}-{}Seed".format(modelType,numSeeds)
    return dfCVEval_seed, yieldDF

def runMultiSeedNestedCV(df,featureDict,numSeeds=5,modelType='RF',maxFeat=10,sortCol=['Epitope-Hits'],AB=False,plot=True,modelKW={}):
    dfEvalYieldModelList = []
    for seed in range(numSeeds):
        if not modelType in ['KNN','QDA']:
            modelKW['random_state'] = seed
        #dfCVEval, yieldDF, modelDict = NestedCVForwardFeatSelect(df,featureDict,maxFeat=maxFeat,sortCol = sortCol,modelType=modelType,AB=AB,plot=plot,modelKW=modelKW)
        dfCVEval, yieldDF, modelDict = NestedCVForwardFeatSelect_multiProcess(df,featureDict,maxFeat=maxFeat,sortCol = sortCol,modelType=modelType,AB=AB,plot=plot,modelKW=modelKW)        
        dfCVEval['Seed'] = seed
        yieldDF['Seed'] = seed
        dfEvalYieldModelList.append((seed,dfCVEval,yieldDF, modelDict))
        if modelType in ['KNN','QDA']:
            break
    dfCVEval_seed, yieldDF = procMultiSeedDFEVal(dfEvalYieldModelList,numSeeds=numSeeds,modelType=modelType,AB=AB,plot=plot)
    return dfCVEval_seed,yieldDF,dfEvalYieldModelList