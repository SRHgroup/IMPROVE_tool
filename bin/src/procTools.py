from __future__ import print_function
import os
import subprocess
import pandas as pd


def makeTempFile(peptList,inDir,peptFile='tempPept.txt'):
    with open(os.path.join(inDir,peptFile),'w') as fo:
        for pept in peptList:
            fo.write('%s\n'%pept)

#def runProcFromList(peptList,inDir,outDir=False,inFile='tempPept.txt',outFile='logoFiles',clean=False,printCMD=False,srcPath=False,**kwargs):
#    makeTempFile(peptList,inDir,inFile)
#    callProc(inDir,outDir=outDir,inFile=inFile,outFile=outFile,clean=clean,printCMD=printCMD,srcPath=srcPath,**kwargs)

def NetMHCIIpanOutput2DF(datDir, inFile,skiprows=11,skipfooter=3,header=False):
    if not header:
        header = ['Seq','Allele','Peptide','Identity','Pos','Core','Core_Rel','1-log50k(aff)','Affinity(nM)','%Rank','Exp_Bind','BindingLevel']
    return pd.read_csv(os.path.join(datDir,inFile),sep="\s+",engine='python',quotechar='#',header=None,names=header,skiprows=skiprows,skipfooter=skipfooter)#read the file, separate columns, define heads

def groupAllelePredNetMHC(df,inDir,peptCol = 'Peptide',outDir=False,inFile='tempPept.txt',outFileTemplate='eval__NetMHCpan4_1__{}__{}.txt',clean=False,printCMD=False,srcPath=False,**kwargs):
    for mhc,dfG in df.groupby('MHC'):
        kwargs['a'] = mhc
        peptList = dfG[peptCol].values
        runNetMHCFromList(peptList,inDir,srcPath=srcPath,outFile = outFileTemplate.format(peptCol,mhc), printCMD=printCMD,**kwargs)

def readNetOutDirApply(inDir,cond = 'eval',header=False,cols=[],skiprows=50,skipfooter=5):
    dfList = []
    for filename in os.listdir(inDir):
        if filename.startswith(cond):
            try:
                df = NetMHCIIpanOutput2DF(inDir,filename,skiprows=skiprows,skipfooter=skipfooter,header=header)
                if len(df)==0:
                    df = NetMHCIIpanOutput2DF(inDir,filename,skiprows=skiprows,skipfooter=skipfooter-1,header=header)
            except:
                print(filename)
                continue
                #sys.exit("Parser errror")
            filenameSplitList = os.path.splitext(filename)[0].split('__')
            for i,col in enumerate(cols):
                df[col] = filenameSplitList[i+1]
            dfList.append(df)
    return pd.concat(dfList)

def readNetMHCpan41(inDir,filename,header=False,skiprows=50,skipfooter=5):
    return NetMHCIIpanOutput2DF(inDir,filename,skiprows=skiprows,skipfooter=skipfooter,header=header)

def groupAlleleRunProc(df,inDir,outDir,peptCol = 'Peptide',mhcCol='MHC',inFile='tempPept.txt',outFileTemplate='eval__NetMHCpan4_1__{}__{}.txt',printCMD=False,srcPath=False,**kwargs):
    for mhc,dfG in df.groupby(mhcCol):
        kwargs['a'] = mhc
        kwargs['i'] = os.path.join(inDir,inFile)
        kwargs['o'] = os.path.join(outDir,outFileTemplate.format(peptCol,mhc))
        peptList = dfG[peptCol].values        
        runProcFromList(peptList, inDir, inFile=inFile, srcPath=srcPath, printCMD=printCMD, **kwargs)
        #break

def runNetMHCFromList(peptList,inDir,outDir=False,inFile='tempPept.txt',outFile='logoFiles',clean=False,printCMD=False,srcPath=False,**kwargs):
    makeTempFile(peptList,inDir,inFile)
    runNetMHCpan(inDir,outDir=outDir,inFile=inFile,outFile=outFile,clean=clean,printCMD=printCMD,srcPath=srcPath,**kwargs)

def runProcFromList(peptList,inDir,inFile='tempPept.txt',printCMD=False,srcPath=False,**kwargs):
    makeTempFile(peptList,inDir,inFile)
    callProc(srcPath, printCMD=printCMD, **kwargs)

def callProc(srcDir,outFile=False,printCMD=False,**kwargs):
    cmd = srcDir
    for key, val in list(kwargs.items()):
        if val=='True':
            cmd+=' -%s'%(key)
        elif val=='False':
            continue
        else:
            cmd+=' -%s %s'%(key, val)
    if outFile:
        cmd = "%s > %s"%(cmd, outFile)
    if printCMD:
        print(cmd)
    return_code = subprocess.call(cmd,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     shell=True
                     )
    #print(return_code)

def CMDHelp(CMD='/Users/birey/Dropbox/2018_TCell_PhD/code/CBS-programs/seq2logo-2.1/Seq2Logo.py'):
    cmd = '{} -h'.format(CMD)
    print(cmd)
    p=subprocess.Popen(cmd,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     shell=True)
    for line in iter(p.stdout.readline, ''):
        if line:
            print(line)
        else:
            break
    p.stdout.close()
    p.kill()

def runNetMHCpan(inDir,outDir=False,inFile='tempPept.txt',outFile='logoFiles',clean=False,printCMD=False,srcPath=False,**kwargs):
    if not srcPath:
        CMDPath = '/Users/annieborch/Documents/programs/netMHCpan-4.1/netMHCpan'
    else:
        CMDPath = srcPath
    if not outDir:
        outDir = inDir
    inPath = os.path.join(inDir,inFile)
    outPath = os.path.join(outDir,outFile)
    cmd = "{} -f {}".format(CMDPath, inPath)
    for key, val in list(kwargs.items()):
        if key=='bg' or key=='blosum':
            cmd+=' --{} {}'.format(key, val)
        elif type(val)==bool and val==True:
            cmd+=' -{}'.format(key)     
        else:
            cmd+=' -{} {}'.format(key, val)
    cmd+=' > {}'.format(outPath)
    if printCMD:
        print(cmd)
    subprocess.call(cmd,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     shell=True)
    if clean:
        os.remove(os.path.join(inDir,inFile))
