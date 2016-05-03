""" filename: ApplicationProfiler
    contents: This program analyzes an input application's address & instruction
    trace to create a characterization (or "profile") that can be used to
    generate synthetic address traces that model the properties of this 
    applicaiton
    
    author: Trevor Gale
    date: 4.14.16"""

from lib.AlphaTree import AlphaTree

import numpy as np
import h5py as h5
import re

def GenerateApplicationProfile(traceFile, outputFile, alphaRatio, blockSize = 512, regEx = None):
    """ GenerateApplicationProfile: this function operates as the main routine
        used to create an application profile from an input address & instruction
        trace
        
        args:
            - traceFile: string indicating the name of the file that contains the
            trace to be analyzed (plain-text)
            - outputFile: string indicating the desired file name for the
            application profile to be stored in
            - alphaRatio: computed as reuse_distances / alpha tree. This value
            affects the number of seperate sets of alpha valeus that will be
            collected to represent the spatial locality of the application's
            memory reference stream. Smaller values have been shown to produce
            more accurate results, but at the cost of more memory usage and a
            larger characterization. The value '0' indicates to use 1 set of
            alpha values for all reuse distances
            - blockSize: desired size of largest cache block to model. Default
            is 512 bytes
            - regEx: Python regular expression to use to pull memory accesses
            the trace. Must not match instruction lines in trace. Defaults to
            <access_type>,<address>. Where access_type is either 'r' or 'w' and
            address is a 4 byte memory address in hexadecimal format e.g. 0x0000fc4e"""
    # TODO: validate input
            
    # if 1 tree for all reuse distances
    if not alphaRatio:
        alphaRatio = float('inf')
        
    # set regular expression
    if not regEx:
        regEx = re.compile("(\D),0x([0-9a-f]+)")
        
    # set mask to pull blockAddress
    blockMask = 2**32 - blockSize
    
    # markov matrix for cycle activity
    activityMarkov = np.zeros((2,2), dtype = np.float)
    previousCycle = 0 # indicates previous cycle's activity
    
    # least-recently used ordered list of all 512-byte blocks accessed
    lruStack = []
    
    # probabilty mass function of all reuse distances
    reusePMF = [0]
        
    # maintains size of the working set
    wsSize = 0
    
    # list of load (read) proportions for each reuse distance
    loadProp = [0]
    lsMap = {'r': 0, 'w':1}
    
    # list of AlphaTree objects to collect alpha values
    alphaTrees = [AlphaTree(blockSize)]
    
    with open(traceFile) as file:
        # TODO: make sure we aren't counting mem instr as seperate cycle?
        for line in file:
            if not re.search(regEx, line): # process inactive cycle
                activityMarkov[previousCycle, 0] += 1
                previousCycle = 0
                continue
            
            # process active cycle
            activityMarkov[previousCycle, 1] += 1
            previousCycle = 1
            
            # seperate memory access into type and location
            memAddress = int(re.search(regEx, line).group(2), 16)
            accessType = re.search(regEx, line).group(1)

            # get block address memAddress            
            memBlock = memAddress & blockMask
            
            # convert access type to numeric representation
            accessType = lsMap[accessType]
                        
            # look up reuse distance of this access
            reuseDist = 0
            while reuseDist < wsSize:
                if lruStack[reuseDist] == memBlock:
                    break
                reuseDist += 1
            
            if reuseDist == wsSize: # if address not previously used
                # process reuse distance
                reusePMF[0] += 1
                reusePMF.append(0)
                wsSize += 1
                
                # update lruStack
                lruStack.insert(0, memBlock)
                
                # increase size of loadProp to match reusePMF
                loadProp.append(0)
                if not accessType: # if load
                    loadProp[0] += 1
                
                # create AlphaTree for new reuse distance, if neccessary
                if alphaRatio != float('inf') and not (wsSize % alphaRatio):
                    alphaTrees.append(AlphaTree(blockSize))
                    
                # update alphaTree
                alphaTrees[0].ProcessAccess(memAddress)
                
                continue
            
            # process reuse distance
            reusePMF[reuseDist + 1] += 1
            
            # update lruStack
            lruStack.remove(memBlock)
            lruStack.insert(0, memBlock)
            
            # process accesst type
            if not accessType: # if load
                loadProp[reuseDist + 1] += 1
            
            # update alphaTree
            alphaTrees[int((reuseDist + 1) / alphaRatio)].ProcessAccess(memAddress)
            
    # TODO: post processing for structures
            
if __name__ == "__main__":
    try:
        GenerateApplicationProfile("traces/testTrace.txt", "test", 0)      
    
    except ValueError as error:
        print "ValueError: " + str(error)
            
