""" filename: TraceGenerator.py
    contents: this script calls the GenerateSyntheticTrace method that
    creates and stream of memory references that models the memory
    performance of the input application or applications
    
    author: Trevor Gale
    date: 3.4.16"""

import h5py as h5
import numpy as np
import ConfigParser
import json
from math import ceil

import sys
import traceback

from lib.AlphaTree import AlphaTree
import lib.TraceFormats as TraceFormats
import lib.PreProcessing as PreProc

# dictionary for all available trace formats
traceFormats = {"STL":TraceFormats.STL, \
    "OVP":TraceFormats.OVP, \
    "Din":TraceFormats.Din}

# usage string
usage_info = "USAGE: python TraceGenerator.py <config_file> \n\
config_file: file specifying the configuration for the trace generator\n\n\
all options for generator must be under header \"[generator]\" \n\
generator options: \n\
\t- traceFile: string indicating the name of the file that contains\n\
\tthe trace to be analyzed (plain-text)\n\n\
\t- traceLength: desired length of the synthetic address\n\
\ttrace (in memory references)\n\n\
\tappProfiles: lists (in brackets, separated by commas, no spaces)\n\
\tof the names of application profiles to model. At least one must \n\
\tbe specified. If len(appProfiles) > 1, the applications profiles\n\
\t are mixed by creating a linear combination with the weights\n\
\tspecified by the \"weights\" parameters. If weights is left as\n\
\tdefault, a uniform distribution is used\n\n\
\t- weights: list (in brackets, separated by commas, no spaces)\n\
\tof the weights of each application. Defaults to evenly weighted\n\
\tapplications\n\n\
\t- formatAccess: name of callback function that is called to print the\n\
\tmemory access. Function must be defined in the lib/TraceFormats and be\n\
\tpresent in the \"traceFormats\" dictionary at the top of this file\n"

# TODO:
# 1. Test mixing 
# 2. Test accuracy with cache model
# LATER
# 1. Tool to generate profiles based on a PMF 
# 2. Print runtime generation details & progress
def GenerateSyntheticTrace(traceFile, traceLength, appProfiles, weights=[], formatAccess=TraceFormats.STL):
    """ GenerateSyntheticTrace: this function takes in application profiles
    generated by the \"ApplicationProfiler\" script and generates a synthetic
    address trace that models the properties of the input applications
    
    args:
        - traceFile: string specifying the name of the file to write the 
        synthetic address trace to (plain-text)
        
        - traceLength: desired length of the synthetic trace (in memory references)
        
        - appProfiles: python list of the names of the application profiles
        to model. At least one must be specified. If len(appProfiles) > 1, the
        applications profiles are mixed by creating a linear combination with 
        the weights specified by the "weights" parameters. If weights is left
        as default, a uniform distribution is used
        
        - weights: python list specifying the weights of each application. 
        Defaults to evenly weighted applications
        
        - formatAccess: callback function that is called to print the memory
        references. Function arguments must be (cycle, accessType, memAddress)"""

    # validate inputs
    if not len(appProfiles):
        raise ValueError("(in GenerateSyntheticTrace) must input >= 1 app profile")
        
    if traceLength <= 0:
        raise ValueError("(in GenerateSyntheticTrace) traceLength must be > 0)")
    
    numProfiles = len(appProfiles)
    if numProfiles > 1 and not(len(weights) == 0 or len(weights) == numProfiles):
        raise ValueError("(in GenerateSyntheticTrace) if len(appProfiles) > 1, len(weights) must be 0 or len(appProfiles)")
    
    # create even weights if weights is left as default
    if len(weights) == 0:
        weights = np.ones(numProfiles)
        
    for i in xrange(numProfiles):
        if weights[i] < 0:
            raise ValueError("(in GenerateSyntheticTrace) weights must be > 0")
        
    # markov model for cycle activity
    blockSize = np.zeros(numProfiles, dtype = np.int)
    
    # open application profiles & find size of largest reusePMF
    numReuseDistances = 0
    for i in xrange(numProfiles):
        appProfiles[i] = h5.File(appProfiles[i])
        
        # get blocksizes
        blockSize[i] = appProfiles[i]['blockSize'][()]
        
        # if current size larger than previous largest rPMF
        if len(appProfiles[i]['reusePMF']) > numReuseDistances:
            numReuseDistances = len(appProfiles[i]['reusePMF'])       
    
    # make sure all blocksizes are the same
    for i in xrange(1, numProfiles):
        if blockSize[i] != blockSize[i-1]:
            raise ValueError("(in GenerateSyntheticTrace) all profiles must have the same blockSize")
    blockSize = blockSize[0] # set blocksize
    
    # build markov model
    activityMarkov = np.zeros((2,2), dtype = np.float)
    PreProc.BuildMarkovModel(appProfiles, weights, activityMarkov)
   
    # create weighted PMF for each reuse distance
    reusePMF = np.zeros(numReuseDistances, dtype = np.float)
    PreProc.BuildReusePMF(appProfiles, weights, reusePMF)

    # create load proportions for each reuse distance
    loadProp = np.zeros(numReuseDistances, dtype = np.float)
    PreProc.BuildLoadProp(appProfiles, weights, loadProp)

    # initialize application's working set
    workingSet = PreProc.BuildWorkingSet(appProfiles).tolist()
    wsSize = len(workingSet)
    lruStack = workingSet
    
    # create alphaForest
    alphaForest = []
    PreProc.BuildAlphaForest(appProfiles, weights, alphaForest, wsSize, blockSize)
        
    # close application profiles
    for i in xrange(numProfiles):
        appProfiles[i].close()  
    
    # open traceFile
    traceFile = open(traceFile, 'w')
    
    # get reference to random generator
    choice = np.random.choice

    # indicates previous cycle's activity
    previousCycle = 0 
    
    # counts unique accesses
    uniqueAddrs = 0
    
    # generation loop
    cycle = -1
    accesses = 0
    while (accesses < traceLength):
        if not choice(2, p=activityMarkov[previousCycle,:]): # if inactive cycle
            # process inactive cycle
            cycle += 1            
            previousCycle = 0
            continue
                    
        # else, active cycle
        cycle += 1
        previousCycle = 1
        accesses += 1
        
        # select reuse distance
        reuseDist = choice(numReuseDistances, p = reusePMF)
        
        # compulsory cache miss
        if not reuseDist:
            # if we run out of addresses, print message and exit
            if uniqueAddrs >= wsSize:
                print "Exiting on cycle %d: cannot exceed size of working set"
                exit()
            
            # select new cache block to reference
            memAddress = lruStack[uniqueAddrs]
            
            # update lru stack
            lruStack.remove(memAddress)
            lruStack.insert(0, memAddress)
           
        else:
            # keep track of unique accesses
            if reuseDist > uniqueAddrs:
                uniqueAddrs += 1
                
            # get block at this reuse distance
            memAddress = lruStack[reuseDist - 1]
            
            # update lruStack
            lruStack.remove(memAddress)
            lruStack.insert(0, memAddress)
            
        # select type of access
        rand = np.random.rand() 
        if rand < loadProp[reuseDist]:
            accessType = 0 # load
        else:
            accessType = 1 # store

        # select 4-byte word address based on alpha values
        blockIndex = workingSet.index(memAddress)
        memAddress = memAddress | alphaForest[blockIndex].GenerateAccess(reuseDist - 1)
        
        # print access
        formatAccess(traceFile, cycle, accessType, memAddress)
        
#
## main function
#
    
if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise IndexError("Invalid number of arguments. Only config file should be specified")
            
        # setup config parser with default args
        config = ConfigParser.RawConfigParser({'weights': [], 'formatAccess': traceFormats['STL']})
        config.read(sys.argv[1])
        
        # pull arguments
        traceFile = config.get('generator', 'traceFile')
        traceLength = int(config.get('generator', 'traceLength'))
        appProfiles = json.loads(config.get('generator', 'appProfiles'))
        weights = json.loads(config.get('generator', 'weights'))
        formatAccess = config.get('generator', 'formatAccess')
        
        GenerateSyntheticTrace(traceFile, traceLength, appProfiles, weights, traceFormats[formatAccess])
    
    except IOError as error:
        print "IOError: " + str(error)
        
    except ValueError as error:
        tb = sys.exc_info()[2]
        traceback.print_tb(tb)
        print "ValueError: ", error
    
    except ConfigParser.NoOptionError as error:
        print "Invalid Args: ", error, "\n"
        print usage_info
    
    except ConfigParser.NoSectionError as error:
        print "Invalid Config: ", error, "\n"
        print usage_info
        
    except KeyError as error:
        print "KeyError: ", error
        print "Invalid application profile: key not found in file"