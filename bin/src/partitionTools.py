import pandas as pd
import datetime

def kmerize(pept,k=9):
    if len(pept)<k:
        raise Exception('Peptide: {} shorter than k: {}, Exiting!'.format(pept,k))
    else:
        return [pept[i:i+k] for i in range(len(pept)-k+1)]

def kmerSet(pept,k=9):
    return set(kmerize(pept,k=k))

def makeRefset(pepts,k=9):
    refSet = set()
    for pept in pepts:
        qSet = kmerSet(pept,k=k)
        refSet = refSet.union(qSet)
    return refSet

def readEpiFiles(datDir,filename,k=9):
    refSet = set()
    with open(os.path.join(datDir,filename),'r') as fh:
        for line in fh:
            line = line.strip()
            qSet = kmerSet(line,k=k)
            refSet = refSet.union(qSet)
    return refSet

def removeOverlap(pept,refSet,k=9):
    qSet = kmerSet(pept,k=k)
    if len(qSet.intersection(refSet))==0:
        return True
    else:
        return False

def findOverlap(df,refSet,peptCol = 'Peptide',k=9):
    return df[~df[peptCol].apply(lambda pept:removeOverlap(pept,refSet,k=k))]

def findNonOverlap(df,refSet,peptCol = 'Peptide',k=9):
    return df[df[peptCol].apply(lambda pept:removeOverlap(pept,refSet,k=k))]

def mapIDX2prot(clust,idx2protDict):
    return [set(clust),set([idx2protDict[c] for c in clust])]

def getIDXprotClustList(df,IDXs,protCol='UID'):
    idx2protDict = dict(zip(range(len(df)),df[protCol].values))
    clustList = [mapIDX2prot(clust,idx2protDict) for clust in IDXs]
    return sorted(clustList,key=lambda x:len(x[1]),reverse=True)


def getKmerClust_prot(clustList,k=9):
    setList = []
    for i,(idxs,prots) in enumerate(clustList):
        overlapFlag = False
        for j in range(len(setList)):
            if len(setList[j][1].intersection(prots))>0:
                setList[j][0] = setList[j][0].union(idxs)
                setList[j][1] = setList[j][1].union(prots)
                overlapFlag = True
                break
        if not overlapFlag:
            setList.append([idxs,prots])
    IDXs,Prots = zip(*setList)
    return [list(idx) for idx in IDXs]

def getKmerClust(peptides,k=9):
    setList = []
    for i,pept in enumerate(peptides):
        kmerSet = kmerSet(pept,k=k)
        overlapFlag = False
        for j in range(len(setList)):
            if len(setList[j][0].intersection(kmerSet))>0:
                setList[j][0] = setList[j][0].union(kmerSet)
                setList[j][1].append(i)
                overlapFlag = True
                break
        if not overlapFlag:
            setList.append([kmerSet,[i]])
    kmerSets,IDXs = zip(*setList)
    return IDXs

def makePartitions(idxList,parts=5):
    partitions = [[] for i in range(parts)]
    idxList = sorted(idxList,key=lambda x:len(x))
    while len(idxList)>0:
        idxs = idxList.pop()
        partitions[0] += idxs
        partitions = sorted(partitions,key=lambda x:len(x))
    return partitions

def CVpartitionsIDXs(idxs,fold=5):
    testIDXs = idxs
    trainIDXs = [[] for x in range(fold)]
    for i in range(fold):
        for j,idx in enumerate(idxs):
            if j!=i:
                trainIDXs[i]+=idx
        #trainIDXs.append([k for k in idxs for j,idx in enumerate(idxs) if j!=i])
    return testIDXs, trainIDXs

def getPartitionColumn(testIDXs):
    partitions = [0]*sum([len(i) for i in testIDXs])
    for i,parts in enumerate(testIDXs):
        print(i,len(parts))
        for j,idx in enumerate(parts):
            partitions[idx] = i
    return partitions

def commonMotifPartitionWrapper(df,fold=5,k=9,peptCol='Peptide',addPartCol=True):
    IDXs = getKmerClust(df[peptCol].values,k=k)
    partitions = makePartitions(IDXs,parts=fold)
    testIDXs,trainIDXs = CVpartitionsIDXs(partitions,fold=fold)
    if addPartCol:
        df['Partition'] = getPartitionColumn(testIDXs)
    return testIDXs,trainIDXs,df

def commonMotifPartitionWrapper_Prot(df,fold=5,k=9,peptCol='Peptide',protCol='UID',addPartCol=True):
    print("Start - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))
    IDXsPept = getKmerClust(df[peptCol].values,k=k)
    print("Common Motif Clustering - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))    
    clustList = getIDXprotClustList(df,IDXsPept,protCol=protCol)
    print("Map IDX to Prot - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))    
    IDXs = getKmerClust_prot(clustList)
    print("Protein Level Clustering - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))        
    partitions = makePartitions(IDXs,parts=fold)
    print("Partitions Made - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))            
    testIDXs,trainIDXs = CVpartitionsIDXs(partitions,fold=fold)
    print("CV folds made - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))                
    if addPartCol:
        df['Partition'] = getPartitionColumn(testIDXs)
    return testIDXs,trainIDXs,df
