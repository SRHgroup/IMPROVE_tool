#!/usr/bin/env python3
import os
import numpy as np

def readBlos2Mat(blosPath):
    lineList = []
    with open(blosPath) as fh:
        for line in fh:
            stripline = line.strip()
            if stripline.startswith("#"):
                continue
            elif stripline.startswith("A"):
                aminoAcids = "".join(stripline.split(" "))
                continue
            lineFloats = [float(num) for num in stripline.split(' ')]
            lineList.append(lineFloats)

    d = len(aminoAcids)
    blosMat = np.zeros((d,d))
    aminoAcids
    for i in range(d):
        for j in range(d):
            if j<=i:
                blosMat[i,j] = lineList[i][j]
            else:
                blosMat[i,j] = lineList[j][i]
    return blosMat, aminoAcids

def genK1(blosMat,beta = 0.11387):
    shape = np.shape(blosMat)
    d = shape[0]
    marg = np.zeros(d)
    k1 = np.zeros(shape)
    for i in range(d):
        Sum = 0
        for j in range(d):
            Sum+=blosMat[i,j]
        marg[i] = Sum
   
    for i in range(d):
        for j in range(d):
            k1[i,j] = blosMat[i,j]/(marg[i]*marg[j])
            k1[i,j] = k1[i,j]**beta
    return k1

def k2_prod(seq1,seq2,start1,start2,k,k1,aminoAcids):
    k2 = 1
    for i in range(k):
        a1 = seq1[i+start1]
        i1 = aminoAcids.find(a1)
        a2 = seq2[i+start2]
        i2 = aminoAcids.find(a2)
        k2 *= k1[i1][i2]
    return k2

def k3_sum(seq1,seq2,k1,aminoAcids,p_min=1,p_max=25):
    k3 = 0
    l1 = len(seq1)
    l2 = len(seq2)
    for k in range(p_min,p_max+1):
        start1 = 0
        while start1 <= l1-k:
            start2 = 0
            while start2 <= l2-k:
                prod = k2_prod(seq1,seq2,start1,start2,k,k1,aminoAcids)
                k3 += prod
                start2 += 1
            start1 += 1                
    return k3

def kernelRunner(seq1,seq2,k1,aminoAcids,p_min=1,p_max=30):
    score11 = k3_sum(seq1,seq1,k1,aminoAcids)
    score22 = k3_sum(seq2,seq2,k1,aminoAcids)
    score12 = k3_sum(seq1,seq2,k1,aminoAcids)
    return score12/((score11*score22)**0.5)

def kernelWrapper(blosPath,dfMerge):
    blosMat,aminoAcids = readBlos2Mat(blosPath)
    k1 = genK1(blosMat)
   # scoreList = []
    dfMerge['SelfSim'] = 'nan'
    for i in dfMerge.index: 
       # print(dfMerge['PeptMut'][i],dfMerge['PeptNorm'][i])
        pep1 = dfMerge['PeptMut'][i]
        pep2 = dfMerge['PeptNorm'][i]
        if pep1 is np.nan or pep2 is np.nan: 
            score = 'nan'
        else: 
            score = kernelRunner(pep1,pep2,k1,aminoAcids)
          #  scoreList.append(score)
            dfMerge['SelfSim'][i] = score
    return dfMerge

   
  #  scoreList = []
  #  for (pept1,pept2) in peptList:
  #      score = kernelRunner(pept1,pept2,k1,aminoAcids)
   #     scoreList.append(score)
    #    dfMerge['SelfSim'] = score
   # return scoreList



