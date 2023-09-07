import os

def defineBuildPath(pathList):
    definedPath = os.path.join(*pathList)
    if not os.path.exists(definedPath):
        os.makedirs(definedPath)
    return definedPath

def catFiles(outFile,inFiles):
    with open(outFile,'w') as fo:
        for inFile in inFiles:
            with open(inFile,'r') as fi:
                for line in fi:
                    fo.write(line)

def readFileGenerator(inPath,splitter='\t'):
	with open(inPath,'r') as fh:
		for line in fh:
			yield line.strip().split(splitter)
            
def clearDirectory(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory,filename))