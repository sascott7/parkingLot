import numpy as np
from yolo import darknet as dn

def createArray(detectionList):
    arr = []
    for line in detectionList:
        if line != []:
            words = line.split()
            if words[0] == "2" or words[0] == "7":
                #if car or truck add to array
                arr.append([float(words[1]), float(words[2])])
    arr = np.asarray(arr)
    return arr

def detect(path, net, meta):
    detectname = bytes(path, "utf-8")
    r = dn.detect(net, meta, detectname)
    return r.splitlines()