#!/bin/bash

# dataFile, outputFile ways, blockSize, cacheSize, writeAllocate
ways=4
blockSize=512
cacheSize=2048
writeAllocate=1
#python ~/Desktop/NUCAR/ADI/Source/ARMCache.py $1 $2 $cacheSize $ways $blockSize $writeAllocate

export PATH=$PATH:/Users/trevorgale/Desktop/DineroIV/d4-7

dineroIV -l1-usize $cacheSize -l1-ubsize $blockSize -l1-uassoc $ways -informat d < $1 
#&> $2
