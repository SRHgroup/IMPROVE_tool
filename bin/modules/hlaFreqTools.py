import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rankEpiTools
import dfTools

import modelTrainEval

def removeLastLetter(typing):
    if typing[-1].isalpha():
        return typing[:-1]
    else:
        return typing

def readFreqTable(directory,loci):
    dfHLA = pd.read_excel(os.path.join(directory,'HLA_{}.xls'.format(loci))).rename(columns={loci:'HLA'})
    dfHLA['Loci'] = loci
    dfHLA['Typing'] = dfHLA['HLA'].apply(removeLastLetter)
    dfHLA['MHC'] = dfHLA['Loci']+dfHLA['Typing']
    return dfHLA

def makeHLApopulation(n,dfA,dfB,dfC,population='EUR_freq'):
    a1 = dfA.sample(n,replace=True,weights=dfA[population])['MHC'].values
    a2 = dfA.sample(n,replace=True,weights=dfA[population])['MHC'].values
    b1 = dfB.sample(n,replace=True,weights=dfB[population])['MHC'].values
    b2 = dfB.sample(n,replace=True,weights=dfB[population])['MHC'].values
    c1 = dfC.sample(n,replace=True,weights=dfC[population])['MHC'].values
    c2 = dfC.sample(n,replace=True,weights=dfC[population])['MHC'].values
    persons = list(zip(a1,a2,b1,b2,c1,c2))
    return persons

def countAlleleIntersectionsInPopulation(population,coveredAlleleSet):
    interSect = {}
    for person in population:
        numInter = len(set(person).intersection(coveredAlleleSet))
        try:
            interSect[numInter]+=1
        except KeyError:
            interSect[numInter]=0
    return interSect

def getMasterHLAFreqTable(freqDir='/Users/birkirreynisson/pac/dat/HLAfreq',loci = ['A','B','C']):
    dfList = []
    for locus in loci:
        df = readFreqTable(freqDir,locus)
        dfList.append(df)
    return pd.concat(dfList)


def getPatientTypingFromDF(df,coveredAlleles,plot=False):
    patientTyping = df.groupby('Patient').apply(lambda dfG: dfG['HLA'].unique()).reset_index().rename(columns={0:'Alleles'})
    patientTyping['HLA-Hits'] = patientTyping['Alleles'].apply(lambda alleles: sum([allele in coveredAlleles for allele in alleles]))
    patientTyping['Dataset'] = patientTyping['Patient'].apply(lambda pat: pat.split('-')[0])
    patientTyping['AlleleNum'] = patientTyping['Alleles'].apply(len)
    if plot:
        sns.countplot(data=patientTyping,x='HLA-Hits')
        plt.show()
    return patientTyping

def updatePatientYields(dfEval,dfYields,hlaHitsRange=range(3)):
    dfList = []
    dfEval2 = dfEval.copy(deep=True)
    for i in hlaHitsRange:
        dfOut = dfEval2[dfEval2['HLA-Hits']>=i]
        patientsHLA_single = dfOut.groupby('Dataset').apply(lambda dfG: dfG['Patient'].nunique()).reset_index().rename(columns={0:'Patient-HLA'})
        patientsHLA_single['HLA-Hits'] = i
        dfList.append(patientsHLA_single)

    patientsHLA = pd.concat(dfList)
    dfYields = dfYields.merge(patientsHLA)
    
    patientsAll = dfEval.groupby('Dataset').apply(lambda dfG: dfG['Patient'].nunique()).reset_index().rename(columns={0:'Patient-All'})
    dfYields = dfYields.merge(patientsAll)

    dfYields['Patient Inclusion%'] = dfYields['Patient-HLA']/dfYields['Patient-All']*100
    dfYields['Patient Inclusion%'] = dfYields['Patient Inclusion%'].apply(lambda hit:round(hit,1))
    dfYields['Patient HitInc%'] = dfYields['Patient-Hits']/dfYields['Patient-HLA']*100
    dfYields['Patient HitInc%'] = dfYields['Patient HitInc%'].apply(lambda hit:round(hit,1))    
    return dfYields

def groupYieldHLAHits(df,yieldKW,hlaHitsRange=range(3),sampling=2):
    yieldDictList = []
    model = '__'.join(df['Model'].unique())
    dfOut = df.copy(deep=True)
    for hlaHits in hlaHitsRange:
        for dataset,dfG in dfOut.groupby('Dataset'):
            dfGout = dfG[dfG['HLA-Hits']>=hlaHits].copy(deep=True)
            #print(hlaHits,dataset)
            #if len(dfGout):
            #    print("GGGGGGGGGGGGGGGGG")
            X = 20 if dataset=='Tesla' else 50
            #print(yieldKW)
            yieldDict = rankEpiTools.yieldsWrapperStats(dfGout,hue=True,targCol=yieldKW['Targ'],predCol=yieldKW['Pred'],plot=yieldKW['Plot'],rank=yieldKW['Rank'],Print=False,X=X)
            #yieldDict = rankEpiTools.yieldsWrapperStats(dfG,hue=True,targCol='Target',predCol='meanPred',plot=False,rank=False,Print=False,X=X)            
            #yieldDict = rankEpiTools.yieldsWrapperStats(dfG,hue=True,targCol='Target',predCol='CombFeat_idx',plot=False,rank=True,Print=False,X=X)            
            yieldDict['Dataset']  = dataset
            yieldDict['HLA-Hits']  = hlaHits
            yieldDict['Model']  = model
            yieldDict['Model+HLA']  = "{}- HLA:{}".format(model,hlaHits)
            yieldDictList.append(yieldDict)
            
            if sampling>0:
                yieldDict_random = modelTrainEval.getRandomSampleYieldDict(dfGout,sampling=sampling).reset_index(drop=True)
                yieldDict_random = dict(list(zip(yieldDict_random.columns.values,yieldDict_random.values[0])))
                yieldDict_random['Dataset']  = dataset
                yieldDict_random['HLA-Hits']  = hlaHits
                yieldDict_random['Model']  = 'Random'
                yieldDict_random['Model+HLA']  = "{}- HLA:{}".format("Random",hlaHits)            
                yieldDictList.append(yieldDict_random)            
            
    return pd.DataFrame(yieldDictList)

def hlaHitYieldsWrapper(dfEval,coveredAlleles,yieldKW,hlaHitsRange=range(3),sampling=2):
    dfOut = dfEval.copy(deep=True)
    patientTyping = getPatientTypingFromDF(dfOut,coveredAlleles)
    dfOut = dfOut.merge(patientTyping[['Patient','HLA-Hits','AlleleNum']])
    dfOut = dfOut[dfOut['AlleleNum']>=0]#Only look at patients with at least 3 HLAs in their typing
    #dfOut['Target'] = dfOut.apply(lambda row: row['Target'] if row['HLA'] in coveredAlleles else 0.0,axis=1)
    dfOutFilt = dfTools.dfColContainsAnyFilter(dfOut,coveredAlleles,'HLA').copy(deep=True)
    dfYields = groupYieldHLAHits(dfOutFilt,yieldKW,hlaHitsRange=hlaHitsRange,sampling=sampling)
    dfYields = updatePatientYields(dfOut,dfYields,hlaHitsRange=hlaHitsRange)
    return dfYields

def sumYieldsAcrossDatasets_sortWrapper(df,modelDicts):
    dfYields_summed = sumYieldsAcrossDatasets(df)
    sortkey = {key:val['Order'] for key,val in modelDicts.items()}
    dfYields_summed['Sortkey'] = dfYields_summed['Model'].map(sortkey)
    dfYields_summed = dfYields_summed.sort_values('Sortkey')
    return dfYields_summed

def sumYieldsAcrossDatasets(df):
    yieldDictList = []
    for model, dfG in df.groupby(['Model']):
        yieldSumDict = dict(dfG[['Epitope-Total','Epitope-Hits','Patient-Resp','Patient-Hits','Patient-Total']].sum(axis=0))
        yieldSumDict['Model'] = model
        yieldDictList.append(yieldSumDict)
    dfYields = pd.DataFrame(yieldDictList)

    dfYields['Epitope Yield%'] = dfYields['Epitope-Hits']/dfYields['Epitope-Total']*100
    dfYields['Epitope Yield%'] = dfYields['Epitope Yield%'].apply(lambda hit:round(hit,1))    
    dfYields['Patient Hit%'] = dfYields['Patient-Hits']/dfYields['Patient-Total']*100
    dfYields['Patient Hit%'] = dfYields['Patient Hit%'].apply(lambda hit:round(hit,1))    
    dfYields['Patient Resp%'] = dfYields['Patient-Resp']/dfYields['Patient-Total']*100
    dfYields['Patient Resp%'] = dfYields['Patient Resp%'].apply(lambda hit:round(hit,1))    
    dfYields['MeanEpiHits'] = dfYields['Epitope-Hits']/dfYields['Patient-Hits']
    dfYields['MeanEpiHits'] = dfYields['MeanEpiHits'].apply(lambda hit:round(hit,2))
    return dfYields

def sumYieldsAcrossDatasets_HLAhits(df):
    yieldDictList = []
    for (hlaHit,model), dfG in df.groupby(['HLA-Hits','Model']):
        yieldSumDict = dict(dfG[['Epitope-Total','Epitope-Hits','Patient-Resp','Patient-Hits','Patient-Total','Patient-HLA','Patient-All']].sum(axis=0))
        yieldSumDict['HLA-Hits'] = hlaHit
        yieldSumDict['Model'] = model
        yieldDictList.append(yieldSumDict)
    dfYields = pd.DataFrame(yieldDictList)
    #dfYields = df
    
    dfYields['Epitope Yield%'] = dfYields['Epitope-Hits']/dfYields['Epitope-Total']*100
    dfYields['Epitope Yield%'] = dfYields['Epitope Yield%'].apply(lambda hit:round(hit,1))    
    dfYields['Patient Inclusion%'] = dfYields['Patient-HLA']/dfYields['Patient-All']*100
    dfYields['Patient Inclusion%'] = dfYields['Patient Inclusion%'].apply(lambda hit:round(hit,1))
    
    dfYields['Patient Hit%'] = dfYields['Patient-Hits']/dfYields['Patient-All']*100
    dfYields['Patient Hit%'] = dfYields['Patient Hit%'].apply(lambda hit:round(hit,1))    
    dfYields['Patient Resp%'] = dfYields['Patient-Resp']/dfYields['Patient-All']*100
    dfYields['Patient Resp%'] = dfYields['Patient Resp%'].apply(lambda hit:round(hit,1))
    
    dfYields['Patient HitInc%'] = dfYields['Patient-Hits']/dfYields['Patient-HLA']*100
    dfYields['Patient HitInc%'] = dfYields['Patient HitInc%'].apply(lambda hit:round(hit,1))    
    dfYields['Patient RespInc%'] = dfYields['Patient-Resp']/dfYields['Patient-HLA']*100
    dfYields['Patient RespInc%'] = dfYields['Patient RespInc%'].apply(lambda hit:round(hit,1))    

    dfYields['MeanEpiHits'] = dfYields['Epitope-Hits']/dfYields['Patient-Hits']
    dfYields['MeanEpiHits'] = dfYields['MeanEpiHits'].apply(lambda hit:round(hit,2))
    
    dfYields['HLA-Hit Inclusion'] = dfYields['HLA-Hits'].apply(str)
    return dfYields

def sumYieldsWrapper(df,coveredAlleles,yieldKW,hlaHitsRange=[0,1,2],sampling=2):
    dfYields = hlaHitYieldsWrapper(df,coveredAlleles,yieldKW,hlaHitsRange=hlaHitsRange,sampling=sampling)
    return sumYieldsAcrossDatasets_HLAhits(dfYields)

def addNewAllelesYieldDict(df,coveredAlleles,newAlleles,yieldKW,hlaHitsRange=[0,1,2],addHLAnum=False,sampling=2):
    if not addHLAnum:
        addHLAnum = len(newAlleles)
    dfOut = df.copy(deep=True)
    dfYields_sum = sumYieldsWrapper(dfOut,coveredAlleles,yieldKW,hlaHitsRange=hlaHitsRange,sampling=sampling)
    dfYields_sum['Total-HLA'] = len(coveredAlleles)
    dfYields_sum['New-HLA'] = ''
    dfList = [dfYields_sum]
    #print(newAlleles[:addHLAs])
    currentAlleles = coveredAlleles
    for allele in newAlleles[:addHLAnum]:
        currentAlleles += [allele]
        #print(currentAlleles)
        dfYields_sum = sumYieldsWrapper(dfOut,currentAlleles,yieldKW,hlaHitsRange=hlaHitsRange,sampling=sampling)
        dfYields_sum['Total-HLA'] = len(currentAlleles)
        dfYields_sum['New-HLA'] = allele
        dfList.append(dfYields_sum)
    dfYields_out = pd.concat(dfList)
    dfYields_out['Model-HLA'] = dfYields_out['Model'] + '-' + dfYields_out['HLA-Hits'].apply(str)
    dfYields_out['Patient-RespHit%'] = dfYields_out['Patient-Hits']/dfYields_out['Patient-Resp']
    return dfYields_out

def sampleCoveredAlleles(df,dfHLA,yieldKW,coveredNum = 25,sampleNum=5,hlaHitsRange=[0,1,2],sampling=2):
    dfList = []
    for sample in range(sampleNum):
        print(sample)
        for numAlleles in range(7,coveredNum,2):
            coveredAlleles = dfHLA.sample(n=numAlleles,weights=dfHLA['EUR_freq'])['MHC'].values
            dfOut = df.copy(deep=True)
            dfYields_sum = sumYieldsWrapper(dfOut,coveredAlleles,yieldKW,hlaHitsRange=hlaHitsRange,sampling=sampling)
            dfYields_sum['Total-HLA'] = len(coveredAlleles)
            dfYields_sum['Sample'] = sample
            dfList.append(dfYields_sum)
    return pd.concat(dfList)