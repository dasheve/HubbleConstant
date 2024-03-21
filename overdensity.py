#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import math
import json
import pandas as pd

path="/home/devesh.yadav/devesh/clustering"

lBox=2500
nBox=128
bS=int(math.floor(lBox/nBox))+1

dataDict={"boxLength":lBox, "numberBoxes":nBox, "boxSize":bS}

data=pd.read_csv(path+"/download1.csv")

# data=data[data["vmax"]>180]
data=data[data["mvir"]>1.0e13]
data=data[data["mvir"]<5.3e15]

#Calculating the density matrix
data["modX"]=(data["x"]//bS).astype(int)
data["modY"]=(data["y"]//bS).astype(int)
data["modZ"]=(data["z"]//bS).astype(int)

thMass=1e14
halosAbove=data[data["mvir"]>thMass]
halosBelow=data[data["mvir"]<thMass]

def gridding(file):
    grid=np.zeros((nBox,nBox,nBox))
    for i, j, k in zip(file["modX"],file["modY"],file["modZ"]):
        grid[i][j][k]=grid[i][j][k]+1
    return grid

gridAll=gridding(data)
gridAbove=gridding(halosAbove)
gridBelow=gridding(halosBelow)

numDensity=haloOverDensity=np.zeros((nBox,nBox,nBox))

def overdensity(grid, filename):
    numDensity=grid/bS**3
    meanNumDensity=np.sum(grid)/lBox**3
    dataDict[filename.split("O")[0]+"MeanDensity"]=meanNumDensity
    haloOverDensity=(numDensity-meanNumDensity)/meanNumDensity
    np.save(path+"/data/"+filename, haloOverDensity)

overdensity(gridAll, "allOverDensity")
overdensity(gridAbove, "aboveOverDensity")
overdensity(gridBelow, "belowOverDensity")

import cosmology as cc
from scipy.integrate import quad

a=cc.cosmology(0.307,0.0,-1.0,0.0,0.0482,0.7,2.726,0.8,0.96,np.log10(8.0),1.0)

def biasCalc(M, z=0):
        return a.bias(M,z)

def numDensity(M, z=0):
        return a.Nplus(M,z)

def numerator(logM):
        M=10**(logM)
        return biasCalc(M)*numDensity(M)*M

def denominator(logM):
        M=10**(logM)
        return numDensity(M)*M

biasAbove=quad(numerator, np.log10(1e14), np.log10(5.3e15))[0]/quad(denominator, np.log10(1e14), np.log10(5.3e15))[0]
biasBelow=quad(numerator, np.log10(1e13), np.log10(1e14))[0]/quad(denominator, np.log10(1e13), np.log10(1e14))[0]
bias=biasBelow/biasAbove
dataDict["bias"]=bias

with open (path+"/data/reqData.json", "w") as outfile:
      json.dump(dataDict, outfile)

