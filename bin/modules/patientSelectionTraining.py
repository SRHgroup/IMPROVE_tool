import itertools
import pandas as pd
import numpy as np


import rankEpiTools
import NNanalysis
import modelTrainEval

def plotLineplotCompare(df1,df2,figPath=False,x='CYT-Threshold',labels = ['','']):
    sns.lineplot(data=df1, x=x,y='AUC0.1',linewidth=4,label = labels[0])
    sns.lineplot(data=df2, x=x,y='AUC0.1',linewidth=4,label = labels[1])
    plt.legend()
    if figPath:
        plt.savefig(figPath.format('AUC0_1'),dpi=600)
    plt.show()

    sns.lineplot(data=df1, x=x,y='Epitope Yield%',linewidth=4,label = labels[0])
    sns.lineplot(data=df2, x=x,y='Epitope Yield%',linewidth=4,label = labels[1])
    plt.legend()
    if figPath:
        plt.savefig(figPath.format('EpitopeYield'),dpi=600)

    plt.show()

    sns.lineplot(data=df1, x=x,y='Patient Hit%',linewidth=4,label = labels[0])
    sns.lineplot(data=df2, x=x,y='Patient Hit%',linewidth=4,label = labels[1])
    plt.legend()
    if figPath:
        plt.savefig(figPath.format('PatientYields'),dpi=600)
    plt.show()

def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())

def filterCYTpatients(df,filterCol ='CYT',quant = 0.25):
    df = df.copy(deep=True)
    #cytThresh = df[filterCol].quantile(quant)
    cytThresh = np.quantile(df[filterCol].unique(),quant)
    df = df[df[filterCol].apply(lambda cyt: cyt >= cytThresh)]
    return df

def filterCYTpatientsNotWork(df,filterCol ='CYT',quant = 0.25):
    df = df.copy(deep=True)
    dfList = []
    for dataset, dfG in df.groupby('Dataset'):
    #cytThresh = df[filterCol].quantile(quant)
        cytThresh = np.quantile(dfG[filterCol].unique(),quant)
        dfGG = dfG[dfG[filterCol].apply(lambda cyt: cyt >= cytThresh)]
        dfList.append(dfGG)
        
    return pd.concat(dfList)


def filterPeptideOutliers(df,outCol='classPred',quant = 0.25):
    df = df.copy(deep=True)
    lowQuant,highQuant=quant,1-quant
    lowThresh = df[outCol].quantile(lowQuant)
    highThresh = df[outCol].quantile(highQuant)
    
    df = df[df.apply(lambda row:not (row['Target']==1.0 and row[outCol]<lowThresh) ,axis=1)]
    df = df[df.apply(lambda row:not (row['Target']==0.0 and row[outCol]>highThresh) ,axis=1)]
    return df


def filterPatientOutliers(df,goodPatientFilter=0.25):
    df = df.copy(deep=True)
    goodPatientFilter = 1.0 - goodPatientFilter
    #Remove Patients with no hits in training data
    dfResponse = df.groupby('Patient')['Target'].apply(sum).reset_index()
    posPatients = dfResponse[dfResponse['Target']>0]['Patient'].values
    df = df[df['Patient'].apply(lambda pat: pat in posPatients)]
    #Rank peptides per patient based on prediction
    df = rankEpiTools.rankByFeaturePerPatient(df,feature='classPred',ascending=False,groupCol='Patient')
    #Generate Master Dataframe with Peptcount, TargetCount, and Mean and Geom-MEan indexing
    dfCount = dfTools.groupCount(df,groupCol='Patient')
    
    dfResponse = df.groupby('Patient')['Target'].apply(sum).reset_index()
    dfPos = df[df['Target']==1.0]
    dfTargetIDX = dfPos.groupby('Patient').apply(lambda dfG: dfG['classPred_idx'].unique()).reset_index().rename(columns={0:'Pos-IDX'})
    dfMeanIDX = dfPos.groupby('Patient')['classPred_idx'].apply(np.mean).reset_index().rename(columns={'classPred_idx':'Mean-IDX'})
    dfGMeanIDX = dfPos.groupby('Patient')['classPred_idx'].apply(geo_mean_overflow).reset_index().rename(columns={'classPred_idx':'GMean-IDX'})
    
    dfTrainStat = dfCount.merge(dfResponse,on='Patient',how='left').merge(dfMeanIDX,on='Patient',how='left').merge(dfGMeanIDX,on='Patient',how='left').merge(dfTargetIDX,on='Patient',how='left')#.merge(dfAUC,on='Patient',how='left')
    dfTrainStat['Rel-GMean-IDX'] = dfTrainStat['GMean-IDX']/dfTrainStat['Count']
    dfTrainStat = dfTrainStat.sort_values('Rel-GMean-IDX')
    
    badPatients = dfTrainStat[dfTrainStat['Rel-GMean-IDX']>goodPatientFilter]['Patient'].values
    
    df = df[df['Patient'].apply(lambda pat: not pat in badPatients and pat in posPatients)]
    
    return df


def TrainingPatientFilter(dfTrain,dfTest,features,targCol='Target',outCol='classPred',modelType='NN',filterCol='CYT',colFilter=0,quantFilter=0.0,patientFilter=0.0,**modelKW):
    
    dfTrain = filterCYTpatients(dfTrain,filterCol=filterCol,quant = colFilter)
    
    model = modelTrainEval.getSKlearnModel(modelType,**modelKW)
    model = modelTrainEval.trainSKlearnModel(model,dfTrain,features,targCol)
    
    if patientFilter>0.0 or quantFilter>0.0:
        dfTrainEval = modelTrainEval.evalSKlearnModel(dfTrain,features,model,outCol=outCol)
        dfTrainEval = filterPeptideOutliers(dfTrainEval,quant = quantFilter)

        dfTrainEval = modelTrainEval.evalSKlearnModel(dfTrainEval,features,model,outCol=outCol)
        dfTrainEval = filterPatientOutliers(dfTrainEval,goodPatientFilter=patientFilter)
        #print(len(dfTrain))
        #print(len(dfTrainEval))        
        
        model = modelTrainEval.getSKlearnModel(modelType,**modelKW)
        model = modelTrainEval.trainSKlearnModel(model,dfTrainEval,features,targCol)
            
    dfOut = modelTrainEval.evalSKlearnModel(dfTest,features,model,outCol=outCol)
    
    return dfOut,model

def applyMeanPred(df,groupCols=['Patient','HLA','PeptMut','Target'],predCol='classPred',outCol='meanPred',multiCol='Seed'):
    df = df.copy(deep=True)
    meanPred = df.groupby(groupCols).apply(lambda dfG: dfG[predCol].mean()).reset_index().rename(columns={0:outCol})
    df = df.merge(meanPred)
    
    dfMean = df.drop_duplicates(groupCols)
    #dfMean = df[df[multiCol]==0]
    #print(dfMean.head())
    #sys.exit()
    return dfMean


def TrainingPatientFilter_CV(df,features,targCol='Target',folds=5,outCol='classPred',modelType='NN',filterCol='CYT',colFilter=0,quantFilter=0.0,patientFilter=0.0,**modelKW):
    modelDict = {'models':[]}
    dfList = []
    for fold in range(folds):
        if True:
            dfTest, dfTrain = rankEpiTools.getTestTrain(df,fold)
            if len(dfTest)==0:#Skip empty fold in inner loop of nested cross validation
                continue
            dfOut,model = TrainingPatientFilter(dfTrain,dfTest,features,targCol=targCol,outCol=outCol,modelType=modelType,colFilter=colFilter,filterCol=filterCol,quantFilter=quantFilter,patientFilter=patientFilter,**modelKW)
        else:
            dfList2 = []
            for i in range(10):
                #print(i)
                dfTest, dfTrain = rankEpiTools.getTestTrain(df,fold)
                dfTrain_pos = dfTrain[dfTrain['Target']==1.0]
                dfTrain_neg = dfTrain[dfTrain['Target']==0.0].sample(n=len(dfTrain_pos),random_state=i)
                dfTrain = pd.concat([dfTrain_pos,dfTrain_neg])
                if len(dfTest)==0:#Skip empty fold in inner loop of nested cross validation
                    continue
                dfOut,model = TrainingPatientFilter(dfTrain,dfTest,features,targCol=targCol,outCol=outCol,modelType=modelType,colFilter=colFilter,filterCol=filterCol,quantFilter=quantFilter,patientFilter=patientFilter,**modelKW)
                dfOut['Seed'] = i

                dfList2.append(dfOut)

            if len(dfTest)==0:#Skip empty fold in inner loop of nested cross validation
                continue
            dfOut = pd.concat(dfList2)
            #print(len(dfOut))
            dfOut = applyMeanPred(dfOut)
            #print(len(dfOut))
            dfOut['classPred'] = dfOut['meanPred']
        dfList.append(dfOut)
        modelDict['models'].append(model)
    return pd.concat(dfList),modelDict


def getYieldDFproc(df):
    yieldDict = rankEpiTools.yieldsWrapperStats(df,predCol='classPred',hue=True,plot=False,rank=False,Print=False)
    yieldDF = pd.DataFrame([yieldDict])
    
    yieldDF['AUC'] = NNanalysis.getAUC_EL(df['Target'],df['classPred'])[2]
    yieldDF['AUC0.1'] = NNanalysis.getAUCThresh(df['Target'],df['classPred'],thresh=0.1)[2]
    return yieldDF

def trainingPatientFilter_yieldDFproc(df,colFilter,quantFilter,patientFilter,modelKW=False):
    yieldDict = rankEpiTools.yieldsWrapperStats(df,predCol='classPred',hue=True,plot=False,rank=False,Print=False)
    if modelKW:
        yieldDict['ModelKW'] = modelKW
    yieldDF = pd.DataFrame([yieldDict])
    
    yieldDF['AUC'] = NNanalysis.getAUC_EL(df['Target'],df['classPred'])[2]
    yieldDF['AUC0.1'] = NNanalysis.getAUCThresh(df['Target'],df['classPred'],thresh=0.1)[2]
    #yieldDF['AUC0.1_mean'] = df.groupby('Patient').apply(lambda dfG: NNanalysis.getAUCThresh(dfG['Target'],dfG['classPred'],thresh=0.1)[2]).fillna(0.0).values.mean()

    yieldDF['CYT-Threshold'] = colFilter
    yieldDF['Quant-Threshold'] = quantFilter
    yieldDF['Patient-Threshold'] = patientFilter
    return yieldDF

def trainingPatientFilterApply(dfTrain,dfEval,features,outCol='classPred',modelType='RF',colFilterRange = range(0,8,2),quantFilterRange=[0.0],patientFilterRange=[0.0],**modelKW):
    dfList = []
    for quantFilter in quantFilterRange:
        for colFilter in colFilterRange:
            for patientFilter in patientFilterRange:
                dfEval_out, model = TrainingPatientFilter(dfTrain,dfEval,features,outCol=outCol,modelType=modelType,colFilter=colFilter,quantFilter=quantFilter,patientFilter=patientFilter,**modelKW)
                yieldDF = trainingPatientFilter_yieldDFproc(dfEval_out,colFilter,quantFilter,patientFilter)
                dfList.append(yieldDF)
    yieldDFMaster = pd.concat(dfList).reset_index()
    return yieldDFMaster

def trainingPatientFilterApply_CV(df,features,outCol='classPred',modelType='RF',filterCol='CYT',colFilterRange = range(0,8,2),quantFilterRange=[0.0],patientFilterRange=[0.0],**modelKW):
    dfList = []
    for quantFilter in quantFilterRange:
        for colFilter in colFilterRange:
            for patientFilter in patientFilterRange:
                dfOut, modelDict = TrainingPatientFilter_CV(df,features,outCol=outCol,modelType=modelType,filterCol=filterCol,colFilter=colFilter,quantFilter=quantFilter,patientFilter=patientFilter,**modelKW)
                yieldDF = trainingPatientFilter_yieldDFproc(dfOut,colFilter,quantFilter,patientFilter)
                dfList.append(yieldDF)
    yieldDFMaster = pd.concat(dfList).reset_index()
    return yieldDFMaster


def trainingPatientFilterApply_nestedCVinner(dfTrain,dfEval,features,folds=5,targCol='Target',outCol='classPred',modelType='RF',filterCol='CYT',colFilterRange = [0.0],quantFilterRange=[0.0],patientFilterRange=[0.0],modelKWlist = [{'random_state':42}],sortParam='AUC0.1'):
    dfList = []
    for quantFilter in quantFilterRange:
        for colFilter in colFilterRange:
            for patientFilter in patientFilterRange:
                for modelKW in modelKWlist:
                    dfTrainOut, modelDict = TrainingPatientFilter_CV(dfTrain,features,folds=folds,outCol=outCol,filterCol=filterCol,modelType=modelType,colFilter=colFilter,quantFilter=quantFilter,patientFilter=patientFilter,**modelKW)
                    yieldDF = trainingPatientFilter_yieldDFproc(dfTrainOut,colFilter,quantFilter,patientFilter,modelKW)
                    dfList.append(yieldDF)
    yieldDFMaster = pd.concat(dfList).reset_index()
    yieldDictTop = yieldDFMaster.sort_values(sortParam,ascending=False).iloc[0,:]#Select best hyperparams
    
    quantFilter = yieldDictTop['Quant-Threshold']
    colFilter = yieldDictTop['CYT-Threshold']
    patientFilter = yieldDictTop['Patient-Threshold']
    modelKW_best = yieldDictTop['ModelKW']
    if True:
        dfEval_out, model = TrainingPatientFilter(dfTrain,dfEval,features,outCol=outCol,modelType=modelType,colFilter=colFilter,quantFilter=quantFilter,patientFilter=patientFilter,**modelKW_best)#Retrain besed on best hyperparams
    else:
        dfList2 = []
        for i in range(20):
            #dfTest, dfTrain = rankEpiTools.getTestTrain(df,fold)
            dfTrain_pos = dfTrain[dfTrain['Target']==1.0]
            dfTrain_neg = dfTrain[dfTrain['Target']==0.0].sample(n=len(dfTrain_pos),random_state=i)
            dfTrain = pd.concat([dfTrain_pos,dfTrain_neg])

            dfEval_out, model = TrainingPatientFilter(dfTrain,dfEval,features,outCol=outCol,modelType=modelType,colFilter=colFilter,quantFilter=quantFilter,patientFilter=patientFilter,**modelKW)
            dfEval_out['Seed'] = i
            dfList2.append(dfEval_out)
        dfEval_out = applyMeanPred(dfEval_out)
        dfEval_out['classPred'] = dfEval_out['meanPred']
    
    return dfEval_out,yieldDictTop,model


def getKeyPermutations(my_dict):
    keys, values = zip(*my_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts

def trainingPatientFilterApply_nestedCVouter(df,features,folds=5,targCol='Target',modelType='RF',outCol='classPred',filterCol='CYT',colFilterRange=[0],quantFilterRange=[0.0],patientFilterRange=[0.0],modelKWlist=[{'random_state':42}],sortParam='AUC0.1'):
    modelKWlist = getKeyPermutations(modelKWlist)
    nestedCVdict = {'dfEval':[],
                    'yieldDict':[],
                    'models':[],
                   }
    for fold in range(folds):
        print(fold)
        dfTest, dfTrain = rankEpiTools.getTestTrain(df,fold)
        if len(dfTest)==0:#Skip empty fold in inner loop of nested cross validation
            continue        
        dfEval_out,yieldDictTop,model = trainingPatientFilterApply_nestedCVinner(dfTrain,dfTest,features,folds=folds,targCol=targCol,outCol=outCol,filterCol=filterCol,modelType=modelType,colFilterRange=colFilterRange,quantFilterRange=quantFilterRange,modelKWlist=modelKWlist,sortParam=sortParam)
        yieldDictTop['Fold'] = fold
        
        for key,value in yieldDictTop['ModelKW'].items():
            yieldDictTop["KW-{}".format(key)] = value
        
        nestedCVdict['dfEval'].append(dfEval_out)
        nestedCVdict['yieldDict'].append(yieldDictTop)
        nestedCVdict['models'].append(model)
    nestedCVdict['dfEval'] = pd.concat(nestedCVdict['dfEval'])
    nestedCVdict['yieldDict'] = pd.DataFrame(nestedCVdict['yieldDict'])
    return nestedCVdict