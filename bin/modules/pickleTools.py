import pickle
import os

def pickleDump(obj,pickleDir,filename):
    path = os.path.join(pickleDir,filename)
    with open(path,'wb') as fh:
        pickle.dump(obj, fh)

def pickleLoad(pickleDir,filename):
    path = os.path.join(pickleDir,filename)
    with open(path,'rb') as fh:
        obj = pickle.load(fh)
    return obj