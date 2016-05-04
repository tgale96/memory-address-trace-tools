""" filename: TraceGenerator.py
    contents: this script calls the GenerateSyntheticTrace method that
    creates and stream of memory references that models the memory
    performance of the input application or applications
    
    author: Trevor Gale
    date: 3.4.16"""

import h5py as h5
import numpy as np

# TODO
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
    
    # open application profiles
    for i in xrange(numProfiles):
        appProfiles[i] = h5.File(appProfiles[i])

        # get blocksizes
        blockSize[i] = appProfiles[i]['blockSize'][()]
        
        # create weighted sum of activity markov models
        activityMarkov = np.add(activityMarkov, weights[i] * appProfiles[i]['activityMarkov'])
        
        # TODO: create rcdf, loadProp, alphaTrees & alphaRatio (use smallest value)

    # make sure all blocksizes are the same
    for i in xrange(1, numProfiles):
        if blockSize[i] != blockSize[i-1]:
            raise ValueError("(in GenerateSyntheticTrace) all profiles must \
                have the same blockSize")
                
    # normalize markov model
    activityMarkov[0][:] /= np.linalg.norm(activityMarkov[0][:], 1)
    activityMarkov[1][:] /= np.linalg.norm(activityMarkov[1][:], 1)
    previousCycle = 0 # indicates previous cycle's activity
    
    # close application profiles
    for i in xrange(numProfiles):
        appProfiles[i].close()

    # generation loop    
    cycle = -1
    while (cycle < traceLength - 1):
        if not np.random.choice(2, p=activityMarkov[previousCycle,:]): # if inactive cycle
            # process inactive cycle
            cycle += 1            
            previousCycle = 0
            continue
        
        # else, active cycle
        cycle += 1
        previousCycle = 1
                
#
## main function
#
    
if __name__ == "__main__":
    # TODO: handle len(appProfiles) with config parser
    # TODO: add more verbose exception printing and more succinct handling

    try:
        GenerateSyntheticTrace("testTrace.txt", 100, ['profiles/testProfile.h5', 'profiles/testProfile2.h5'])
    
    except IOError as error:
        print "IOError: " + str(error)
#    
#    except ValueError as error:
#        print "ValueError: " + str(error)
#    