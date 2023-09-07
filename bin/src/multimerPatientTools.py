import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
###
import procTools
import physiochemical_properties
import kernelSim

homeDir = '.'


def MutPos(notation):
    mutPos = []
    if type(notation)==float:
        return mutPos
    for i, notAA in enumerate(notation):
        if notAA!='.':
            mutPos.append(i+1)
    return mutPos

def MutNotation2WT(mutant,notation):
    wt = []
    for i,mutAA in enumerate(mutant):
        if notation[i]=='.':
            wt.append(mutAA)
        else:
            wt.append(notation[i])
    return ''.join(wt)
    
def MutPosCompare(peptMut,peptNorm):
    mutPos = []
    for i, normAA in enumerate(peptNorm):
        if normAA!=peptMut[i]:
            mutPos.append(i+1)
    return mutPos

def Simple_MHCnotation(mhc):
    return mhc.replace('*','').replace(':','').replace('HLA-','')

def NetMHCpan_MHCnotation(mhc):
    return "HLA-{}".format(mhc.replace('*',''))


def NetMHCpan_MHCnotation(mhc):
    return "HLA-{}:{}".format(mhc[:3],mhc[3:])

def ShortHLATyping(mhc):
    mhc = mhc.replace('HLA-','')
    mhc = mhc.replace(':','')
    mhc = mhc.replace('*','')
    return mhc

def fillNAsample(df,feature):
    dfNA = df[df[feature].isna()]
    dfNotNA = df[df[feature].notna()]
    
    dfNA[feature] = dfNotNA.sample(n=len(dfNA))[feature].values
    return pd.concat([dfNA,dfNotNA])

def mergeSourceDFwithNetMHCpan_stab(df,dfNet_mut,dfNet_wt,dfNet_stab):
    colsMut = ['Allele','Peptide','Core','Of','Gp','Gl','Ip','Il','Score_EL','%Rank_EL','Score_BA','%Rank_BA','Aff(nM)']
    colsWT = ['Allele','Peptide','Score_EL','%Rank_EL','Score_BA','%Rank_BA','Aff(nM)']
    
    dfSel_mut_pred = df.merge(dfNet_mut[colsMut],left_on=['MHC','PeptMut'],right_on=['Allele','Peptide'])
    dfSel_mut_wt_pred = dfSel_mut_pred.merge(dfNet_wt[colsWT],left_on=['Allele','PeptNorm'],right_on=['Allele','Peptide'],suffixes = ('_mut','_wt'))
    try:
        dfSel_mut_wt_pred_nonRed = dfSel_mut_wt_pred.drop_duplicates(['PeptMut','HLA','Patient','Target'])
    except KeyError:
        dfSel_mut_wt_pred_nonRed = dfSel_mut_wt_pred.drop_duplicates(['PeptMut','HLA','Patient'])

    colStab = ['Allele','Peptide','Pred','Thalf(h)','%Rank_Stab']
    dfSel_net41_stab = dfSel_mut_wt_pred_nonRed.merge(dfNet_stab[colStab],left_on=['PeptMut','Allele'],right_on=['Peptide','Allele'])
    try:
        dfSel_return = dfSel_net41_stab.drop_duplicates(['PeptMut','Allele','Patient','Target'])
    except KeyError:
        dfSel_return = dfSel_net41_stab.drop_duplicates(['PeptMut','Allele','Patient'])
    return dfSel_return

def clearDirectory(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory,filename))

def defineBuildPath(pathList):
    definedPath = os.path.join(*pathList)
    if not os.path.exists(definedPath):
        os.makedirs(definedPath)
    return definedPath

def predReadWritePRIME(dfSel,predDir,utilsDir_prime,dataSet='BC',outFileTemplate = 'eval__PRIME__{}__{}.txt',resFileTemplate= 'prediction_{}_PRIME.tsv',tmpDir=False, utilsDir = '/Users/annieborch/Documents/programs/'):
    if tmpDir:
        ## This is a hack to get around the problem of PRIME not supporting filenames with spaces
        outDir = predDir        
        predDir=tmpDir
        clearDirectory(tmpDir)
    else:
        outDir = predDir
    
    print("Run PRIME")
    print("Size of input: {}".format(len(dfSel)))
    MixMHCpred = os.path.join(utilsDir,'MixMHCpred-master/MixMHCpred')
    kwargs_prime = {'mix':MixMHCpred}
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

def predReadWriteNetMHCpan_41(dfSel,predDir,utilsDir,dataSet = 'BC',WT=True,outFileTemplate = 'prediction_{}_NetMHCpan41_{}.tsv',**kwargs_net41):
    
    #Prepare parameters
    print("Run NetMHCpan 4.1")
    print("Size of input: {}".format(len(dfSel)))
    wtMut = 'wt' if WT else 'mut'
    peptCol = 'PeptNorm' if WT else 'PeptMut'
    predDir_out = os.path.join(predDir,wtMut)
    
    #Predict binding
    procTools.groupAllelePredNetMHC(dfSel,predDir_out,peptCol = peptCol,inFile='tempPept.txt',srcPath=utilsDir,clean=False,printCMD=False,**kwargs_net41)
    
    #Read prediction outputs from directory
    header_net41 = ['Pos','MHC','Peptide','Core','Of','Gp','Gl','Ip','Il','Icore','Identity','Score_EL','%Rank_EL','Score_BA','%Rank_BA','Aff(nM)','Dummy','BindLevel']
    evalDF_out = procTools.readNetOutDirApply(predDir_out,header=header_net41,cols=['Predictor','PeptType','Allele'])    
    readMask = evalDF_out.isna().sum(axis=1)<len(header_net41)-1
    evalDF_out = evalDF_out[readMask]
    
    #dfRead = readNetOutDirApply(predDir_net41,header=header_net41)
    #readMask = evalDF_out.isna().sum(axis=1)<len(header_net41)-1
    #dfRead = dfRead[readMask]

    print("Size of NetMHCpan output {}: {}".format(wtMut,len(evalDF_out)))
    evalDF_out.to_csv(os.path.join(predDir,'..',outFileTemplate.format(dataSet,wtMut)),sep='\t',index=False)
    
    return evalDF_out

def predReadWriteNetMHCstabPan(dfSel,predDir,utilsDir,dataSet = 'BC',resFileTemplate = 'prediction_{}_NetMHCstabPan.tsv',lenCol='PeptLen'):
    # NetMHCstabPan must be run on a peptide list of single length
    print("Run NetMHCstabPan")
    print("Size of input: {}".format(len(dfSel)))
    for l, dfG in dfSel.groupby(lenCol):
        kwargs_netStab = {'p':True}
        outFileTemplate = 'eval__NetMHCstabpan__{}__{}__'+str(l)+'.txt'
        procTools.groupAllelePredNetMHC(dfG,predDir,peptCol = 'PeptMut',outFileTemplate = outFileTemplate,srcPath=utilsDir,clean=False,printCMD=False,**kwargs_netStab)

    header_netStab = ['Pos','MHC','Peptide','Identity','Pred','Thalf(h)','%Rank_Stab','Dummy','BindLevel']
    evalDF_netStab = procTools.readNetOutDirApply(predDir,header=header_netStab,cols=['Predictor','PeptType','Allele','PeptideLength'])

    print("Size of NetMHCstabPan output: {}".format(len(evalDF_netStab)))
    evalDF_netStab.to_csv(os.path.join(predDir,'..',resFileTemplate.format(dataSet)),sep='\t',index=False)
    return evalDF_netStab

def mergePrepFeatures_noFlurry(dfNet,dfPrime,peptCol='PeptMut',mhcCol='HLA',plot=True):
    dfMerge = dfNet.merge(dfPrime,on=[peptCol,mhcCol])
    #dfNetFlu = dfNetFlu.drop_duplicates([peptCol,'Allele','Patient','Target'])
    # There is a problem with repeated peptides in the dataframe, i.e. the same patient, HLA, peptide sets, but with different Target value
    # Remove this redundancy and ambiguity by saying that it a peptide is EVER measure positive, it should only receive a positive target
    try:
        dfMerge = dfMerge.sort_values('Target',ascending=False)    
    except KeyError:
        pass
    dfMerge = dfMerge.drop_duplicates([peptCol,mhcCol,'Patient'])
    #dfMerge = dfMerge.drop_duplicates()
    dfMerge['Agrotopicity'] = dfMerge['%Rank_EL_mut']/dfMerge['%Rank_EL_wt']
   # dfMerge['Expr/EL_41'] = dfMerge['Expr']/dfMerge['%Rank_EL_mut']
    #dfMerge['CoreNonAnchor'] = dfMerge['Core'].apply(lambda core: core[3:-1])
    #Naive way of getting only TCR facing residues
    dfMerge['CoreNonAnchor'] = dfMerge['Core'].apply(lambda core: core[3:-1])
    #Include into TCR facing those residues that bulge out of the MHC binding core    
    dfMerge['CoreNonAnchor'] = dfMerge['CoreNonAnchor'] + dfMerge.apply(lambda row: row['PeptMut'][row['Gp']:row['Gp']+row['Gl']],axis =1)

    dfMerge['Loci'] = dfMerge[mhcCol].apply(lambda hla: hla[0])
    dfMerge = dfMerge.sort_values(mhcCol)
    # if plot and 'Target' in dfMerge.columns.values:
    #     plotDatasetBarplots(dfMerge,targCol='Target')
    # else:
    #     pass
    dfMerge = addHydrophobHelixCompFeatures(dfMerge)
    return dfMerge


def mergePrepFeatures(dfNet,dfFlurry,dfPrime,peptCol='PeptMut',plot=True):
    dfNetFlu = dfNet.merge(dfFlurry,on=[peptCol,'HLA'])
    #dfNetFlu = dfNetFlu.drop_duplicates([peptCol,'Allele','Patient','Target'])
    # There is a problem with repeated peptides in the dataframe, i.e. the same patient, HLA, peptide sets, but with different Target value
    # Remove this redundancy and ambiguity by saying that it a peptide is EVER measure positive, it should only receive a positive target
    dfNetFlu = dfNetFlu.sort_values('Target',ascending=False)    
    dfNetFlu = dfNetFlu.drop_duplicates([peptCol,'HLA','Patient'])
    dfMerge = dfNetFlu.merge(dfPrime,on=[peptCol,'HLA'])
    dfMerge = dfMerge.drop_duplicates()
    dfMerge['Agrotopicity'] = dfMerge['%Rank_EL_mut']/dfMerge['%Rank_EL_wt']
   # dfMerge['Expr/EL_41'] = dfMerge['Expr']/dfMerge['%Rank_EL_mut']
    #Naive way of getting only TCR facing residues
    dfMerge['CoreNonAnchor'] = dfMerge['Core'].apply(lambda core: core[2:-1])
    #Include into TCR facing those residues that bulge out of the MHC binding core    
    dfMerge['CoreNonAnchor'] = dfMerge['CoreNonAnchor'] + dfMerge.apply(lambda row: row['PeptMut'][row['Gp']:row['Gp']+row['Gl']],axis =1)
    dfMerge['Loci'] = dfMerge['HLA'].apply(lambda hla: hla[0])
    if plot:
        plotDatasetBarplots(dfMerge,targCol='Target')
    return dfMerge

def runFeatureGeneration_NetMHCpan_stab_prime_molecular(df,predDir,dataSet="my_dat",utilsDir = '/Users/annieborch/Documents/programs/',tmpDir = '/Users/annieborch/Documents/programs/',runPred=True,plot=True,clean=False):
    
    predDir_net41 = os.path.join(predDir,'netmhcpan41')
   # predDir_net40 = os.path.join(predDir,'netmhcpan40')
    predDir_netStab = os.path.join(predDir,'netmhcstabpan')
  #  predDir_prime = os.path.join(predDir,'PRIME')
    # prime 1.0
    predDir_prime = os.path.join(predDir,'PRIME-1.0')

    utilsDir_net41 = os.path.join(utilsDir,'netMHCpan-4.1','netmhcpan')
  #  utilsDir_net41 = os.path.join(utilsDir,'netMHCpan-4.0','netmhcpan')
    utilsDir_netStab = os.path.join(utilsDir,'netMHCstabpan-1.0','netMHCstabpan')
   # utilsDir_prime = os.path.join(utilsDir,'PRIME','PRIME')
    # prime 1.0
    utilsDir_prime = os.path.join(utilsDir,'PRIME-1.0','PRIME')

    if clean:
        for predDir in [os.path.join(predDir_net41,'mut'),os.path.join(predDir_net41,'wt'),predDir_netStab,predDir_prime]:
            clearDirectory(predDir)

    if runPred:
        ###NetMHCpan_41
        kwargs_net41 = {'p':True,'BA':True}
        evalDF_net41_mut = predReadWriteNetMHCpan_41(df,predDir_net41,utilsDir_net41,dataSet = dataSet,WT=False,**kwargs_net41)
        evalDF_net41_wt = predReadWriteNetMHCpan_41(df,predDir_net41,utilsDir_net41,dataSet = dataSet,WT=True,**kwargs_net41)
        ###NetMHCstabPan
        evalDF_netStab = predReadWriteNetMHCstabPan(df,predDir_netStab,utilsDir_netStab,dataSet = dataSet)
        ###PRIME
        #dfPrime = predReadWritePRIME(df,predDir_prime,utilsDir_prime,dataSet = dataSet,tmpDir=tmpDir)

        dfPrime = predReadWritePRIME(df,predDir_prime,utilsDir_prime,dataSet = dataSet,tmpDir=False, utilsDir = utilsDir)
    else:
        evalDF_net41_mut = pd.read_csv(os.path.join(predDir,'prediction_{}_NetMHCpan41_mut.tsv'.format(dataSet)),sep='\t')
        evalDF_net41_wt = pd.read_csv(os.path.join(predDir,'prediction_{}_NetMHCpan41_wt.tsv'.format(dataSet)),sep='\t')
        #NetMHCstabPan
        evalDF_netStab = pd.read_csv(os.path.join(predDir,'prediction_{}_NetMHCstabPan.tsv'.format(dataSet)),sep='\t')
        #PRIME
        dfPrime = pd.read_csv(os.path.join(predDir,'prediction_{}_PRIME.tsv'.format(dataSet)),sep='\t')
    #print(evalDF_net41_mut.head())
    
   
    dfNet = mergeSourceDFwithNetMHCpan_stab(df,evalDF_net41_mut,evalDF_net41_wt,evalDF_netStab)
    #dfNet = mergeSourceDFwithNetMHCpan_stab(df,evalDF_net41_mut,evalDF_net41_wt)
   
    dfMerge = mergePrepFeatures_noFlurry(dfNet,dfPrime,peptCol='PeptMut',plot=plot)

    blosFile = "data/matrices/blosum62.qij"    
    blosPath = os.path.join(homeDir,blosFile)
    dfMerge = kernelSim.kernelWrapper(blosPath,dfMerge)

    physiochemical_properties.calculate_and_save_properties(dfMerge, 'PeptMut', 'mw', 'aro', 'inst','helix','cys_red','pI')
    


    return dfMerge

    
def mapDictionaries(d1,d2):
    d12 = {}
    for key,value in d2.items():
        d12[d1[key]] = value
    return d12

# These values are from Expacy: https://web.expasy.org/protscale/

Three2One = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D',
            'Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
            'His':'H','Ile':'I','Leu':'L','Lys':'K',
            'Met':'M','Phe':'F','Pro':'P','Ser':'S',
            'Thr':'T','Trp':'W','Tyr':'Y','Val':'V',
            '-':'-'}

hydrophKyteDool={'Ala':  1.800,'Arg': -4.500,'Asn': -3.500,'Asp': -3.500,
                'Cys':  2.500,'Gln': -3.500,'Glu': -3.500,'Gly': -0.400,
                'His': -3.200,'Ile':  4.500,'Leu':  3.800,'Lys': -3.900,
                'Met':  1.900,'Phe':  2.800,'Pro': -1.600,'Ser': -0.800,
                'Thr': -0.700,'Trp': -0.900,'Tyr': -1.300,'Val':  4.200,
                '-':-0.49}

helixLevitt = {'Ala':1.290,'Arg':0.960,'Asn':0.900,'Asp':1.040,
                'Cys':1.110,'Gln':1.270,'Glu':1.440,'Gly':0.560,
                'His':1.220,'Ile':0.970,'Leu':1.300,'Lys':1.230,
                'Met':1.470,'Phe':1.070,'Pro':0.520,'Ser':0.820,
                'Thr':0.820,'Trp':0.990,'Tyr':0.720,'Val':0.910,
              '-':1.305}

HydrophKyteDool = mapDictionaries(Three2One,hydrophKyteDool)
HelixLevitt = mapDictionaries(Three2One,helixLevitt)
#HelixDeleage = mapDictionaries(Three2One,helixDeleage)


def mapPeptMean(pept,mapper):
    return sum([mapper[AA] for AA in pept])/len(pept)

def genHydrophHelixFeats(df,peptCol='PeptMut',coreCol='Core',coreNoAncCol='CoreNonAnchor'):
    df['MeanHydroph'] = df[peptCol].apply(lambda pept: mapPeptMean(pept,HydrophKyteDool))
    df['MeanHydroph_core'] = df[coreCol].apply(lambda pept: mapPeptMean(pept,HydrophKyteDool))
    df['MeanHydroph_coreNoAnc'] = df[coreNoAncCol].apply(lambda pept: mapPeptMean(pept,HydrophKyteDool))
    return df

def getPeptAAcomp(pept,aminos='ARNDCQEGHILKMFPSTWYV'):
    return {aa:pept.count(aa)/len(pept) for aa in aminos}

def meanComps(compList,AminoAcids='ARNDCQEGHILKMFPSTWYV'):
    summaryComp = {aa:0 for aa in AminoAcids}
    for comp in compList:
        for aa in AminoAcids:
            summaryComp[aa]+=comp[aa]

def getComps(df):
    df['peptComp'] = df['PeptMut'].apply(lambda pept: getPeptAAcomp(pept))
    df['coreComp'] = df['Core'].apply(lambda pept: getPeptAAcomp(pept))
    df['coreNoAncComp'] = df['CoreNonAnchor'].apply(lambda pept: getPeptAAcomp(pept))
    
    
    return df

def getPeptHydroPhComp(pept,aminos):
    return sum([pept.count(aa) for aa in aminos])/len(pept)

def genHydrophCompFeat(df,hydroPhob='CFVHWILM',peptCol='PeptMut',coreCol='Core',coreNoAncCol='CoreNonAnchor'):
    df['HydroPhobRatio_coreNoAnchor'] = df[coreNoAncCol].apply(lambda pept: getPeptHydroPhComp(pept,hydroPhob))
    return df

def getAminoCompSum(compDict,aminos):
    return sum([compDict.get(amino,0.0) for amino in aminos])

AApropDict  = {
            'Prop_Small':['A','B','C','D','G','N','P','S','T','V'],
            'Prop_Aromatic':['F','H','W','Y'],
            'Prop_Basic':['H','K','R'],
            'Prop_Acidic':['B','D','E','Z'],
            }


def getAApropertyFreqSum(df,propDict=AApropDict,seqCol = 'CoreNonAnchor'):
    for key,value in propDict.items():
        df[key] = df[seqCol].apply(lambda seq: getAminoCompSum(getPeptAAcomp(seq),value))
    return df


def addHydrophobHelixCompFeatures(df,plot=False):
    df = genHydrophHelixFeats(df)
    df = getComps(df)
    df = genHydrophCompFeat(df)
    df = getAApropertyFreqSum(df)
    if plot:
        plotEpitopeCompEnrich(df)
    return df






