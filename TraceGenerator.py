""" filename: TraceGenerator.py
    contents: this script calls the GenerateSyntheticTrace method that
    creates and stream of memory references that models the memory
    performance of the input application or applications
    
    author: Trevor Gale
    date: 3.4.16"""

import h5py as h5
import numpy as np

# TODO:
# 1. Generator program (generate list of numbers 1- 2**32/blocksize & remove when selected for unique blocks)
# 1.5 additions to AlphaTree to allow for keeping track of access history without counting, & selecting blocks baed on alphas
# 2. STL printing callback & error handling for incorrect input
# 3. Interface & usage info
# 4. Testing
# 5. Tool to generate profiles based on a PMF 
# 6. Provide indication that program is running (% or just a statement)
def GenerateSyntheticTrace(traceFile, traceLength, appProfiles, weights=[], traceFormat='STL'):
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
        
        - traceFormat: callback function that is called to print the memory
        references. Function arguments must be (cycle, accessType, memAddress)"""

    # validate inputs
    numProfiles = len(appProfiles)
    if numProfiles > 1 and not(len(weights) == 0 or len(weights) == numProfiles):
        raise ValueError("(in GenerateSyntheticTrace) if len(appProfiles) > 1, \
            len(weights) must be 0 or len(appProfiles)")
    
    # create even weights if weights is left as default
    if len(weights) == 0:
        weights = np.ones(numProfiles)
        
    # markov model for cycle activity
    activityMarkov = np.zeros((2,2), dtype = np.float)
    blockSize = np.zeros(numProfiles, dtype = np.int)
    
    # open application profiles, get size of largest rPMF, create markov model
    numReuseDistances = 0
    for i in xrange(numProfiles):
        appProfiles[i] = h5.File(appProfiles[i])
        
        # get blocksizes
        blockSize[i] = appProfiles[i]['blockSize'][()]
        
        # create weighted sum of activity markov models
        activityMarkov = np.add(activityMarkov, weights[i] * appProfiles[i]['activityMarkov'])
        
        # if current size larger than previous largest rPMF
        if len(appProfiles[i]['reusePMF']) > numReuseDistances:
            numReuseDistances = len(appProfiles[i]['reusePMF'])
        
        # TODO: create alphaTrees & alphaRatio (use smallest value)
    
    # make sure all blocksizes are the same
    for i in xrange(1, numProfiles):
        if blockSize[i] != blockSize[i-1]:
            raise ValueError("(in GenerateSyntheticTrace) all profiles must \
                have the same blockSize")
    blockSize = blockSize[0] # set blocksize
    
    # create weighted PMF & load proportions for each reuse distance
    reusePMF = np.zeros(numReuseDistances, dtype = np.float)
    loadProp = np.zeros(numReuseDistances, dtype = np.float)
    for i in xrange(numProfiles):
        temp = np.asarray(appProfiles[i]['reusePMF'])
        temp.resize(numReuseDistances)
        reusePMF = np.add(reusePMF, temp * weights[i])
        
        loadProp = np.add(loadProp, appProfiles[i]['loadProp'])
    
                
    # normalize markov model
    activityMarkov[0][:] /= np.linalg.norm(activityMarkov[0][:], 1)
    activityMarkov[1][:] /= np.linalg.norm(activityMarkov[1][:], 1)
    previousCycle = 0 # indicates previous cycle's activity
    
    # normalize reusePMF
    reusePMF /= np.linalg.norm(reusePMF, 1) 
    
    # least recently used ordered stack of memory references
    lruStack = []
    workingSet = range(2**32 / blockSize)
    
    # normalize loadProp
    weightSum = 0
    for i in xrange(numProfiles):
        weightSum += weights[i]
    loadProp /= weightSum
    
    # close application profiles
    for i in xrange(numProfiles):
        appProfiles[i].close()
    
    # get reference to random generator
    choice = np.random.choice
    
    # generation loop
    cycle = -1
    while (cycle < traceLength - 1):
        if not choice(2, p=activityMarkov[previousCycle,:]): # if inactive cycle
            # process inactive cycle
            cycle += 1            
            previousCycle = 0
            continue
                    
        # else, active cycle
        cycle += 1
        previousCycle = 1
        
        # select reuse distance
        reuseDistance = choice(numReuseDistances, p = reusePMF)
        if (not reuseDistance) or ((reuseDistance-1) >= len(lruStack)): # compusory cache miss
            # select new cache block to reference
            memAddress = workingSet[np.random.choice(len(workingSet))]
            
            # update working set
            workingSet.remove(memAddress)
            
            # convert to block and update lru stack
            memAddress *= blockSize
            lruStack.insert(0, memAddress)
            
            # make sure set for ld/str decision
            reuseDistance = 0            
        else:
            # get block at this reuse distance
            memAddress = lruStack[reuseDistance - 1]
            
            # update lruStack
            lruStack.remove(memAddress)
            lruStack.insert(0, memAddress)
            
        # select type of access
        rand = np.random.rand() 
        if rand < loadProp[reuseDistance]:
            accessType = 0 # load
        else:
            accessType = 1 # store

        # select 4-byte word address based on alpha values
                
#
## main function
#
    
if __name__ == "__main__":
    # TODO: handle len(appProfiles) with config parser
    # TODO: add more verbose exception printing and more succinct handling

    try:
        GenerateSyntheticTrace("testTrace.txt", 10000, ['profiles/testProfile.h5', 'profiles/testProfile2.h5'])
    
    except IOError as error:
        print "IOError: " + str(error)
#    
#    except ValueError as error:
#        print "ValueError: " + str(error)
#    