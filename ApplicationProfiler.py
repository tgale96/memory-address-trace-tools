""" filename: ApplicationProfiler
    contents: This program analyzes an input application's address & instruction
    trace to create a characterization (or "profile") that can be used to
    generate synthetic address traces that model the properties of this 
    applicaiton
    
    author: Trevor Gale
    date: 3.3.16"""

import ConfigParser
import numpy as np
import h5py as h5
import re

import sys
import traceback

from lib.AlphaTree import AlphaTree

# usage string
usage_info = "USAGE: python ApplicationProfiler.py <config_file> \n\
config_file: file specifying the configuration for the application profiler\n\n\
all options for profiler must be under header \"[profiler]\" \n\
profiler options: \n\
\t- traceFile: string indicating the name of the file that contains\n\
\tthe trace to be analyzed (plain-text)\n\n\
\t- outputFile: string indicating the name of the file to save the\n\
\tapplication profile to (automatically appends \".h5\")\n\n\
\t- alphaRatio: ratio of reuse_distances / alpha tree. This value\n\
\taffects the number of seperate sets of alpha values that will be\n\
\tcollected to represent the spatial locality of the application's\n\
\tmemory reference stream. Smaller values have been shown to produce\n\
\tmore accurate results, but at the cost of more memory usage and a\n\
\tlarger characterization. The default value '0' indicates to use 1\n\
\tset of alpha values for all reuse distances\n\n\
\t- blockSize: size of the largest cache block to model(in bytes).\n\
\tdefault is 512 bytes\n\n\
\t- regEx: Python regular expression to use to pull memory accesses\n\
\tthe trace. Must not match instruction lines in trace. Defaults to\n\
\t<access_type>,<address>. Where access_type is either 'r' or 'w' and\n\
\taddress is a 4 byte memory address in hexadecimal format e.g.\n\
\t0x0000fc4e \n\n\
Example configurations can be found in the \"examples\" directory"

# TODO: fix regEx feature and lsMap (only works if groups are positioned correctly in regex)
def GenerateApplicationProfile(traceFile, outputFile, alphaRatio = 0, blockSize = 512, regEx = None):
    """ GenerateApplicationProfile: this function operates as the main routine
        used to create an application profile from an input address & instruction
        trace
        
        args:
            - traceFile: string indicating the name of the file that contains the
            trace to be analyzed (plain-text)
            
            - outputFile: string indicating the desired file name for the
            application profile to be stored in (automatically append ".h5" to
            filename)
            
            - alphaRatio: computed as reuse_distances / alpha tree. This value
            affects the number of seperate sets of alpha values that will be
            collected to represent the spatial locality of the application's
            memory reference stream. Smaller values have been shown to produce
            more accurate results, but at the cost of more memory usage and a
            larger characterization. The default value '0' indicates to use 1 set 
            of alpha values for all reuse distances
            
            - blockSize: desired size of largest cache block to model. Default
            is 512 bytes
            
            - regEx: Python regular expression to use to pull memory accesses
            the trace. Must not match instruction lines in trace. Defaults to
            <access_type>,<address>. Where access_type is either 'r' or 'w' and
            address is a 4 byte memory address in hexadecimal format e.g. 
            0x0000fc4e"""
    # validate inputs
    if alphaRatio < 0:
        raise ValueError("(in GenerateApplicationProfile) alphaRatio must be non-negative integer")
        
    if blockSize % 2 or blockSize < 8:
        raise ValueError("(in GenerateApplicationProfile) blockSize must be power of 2 >= 8")

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
        for line in file:
            
            # process inactive cycle
            if not re.search(regEx, line):
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
        
    # normalize load proprotions
    for i in xrange(len(loadProp)):
        if reusePMF[i]: # if non-zero
            loadProp[i] /= float(reusePMF[i])
    
    # normalize reuse PMF
    reusePMF /= np.linalg.norm(reusePMF, 1)
    
    # remove double counted inactive cycles
    activityMarkov[0][0] -= activityMarkov[0][1]

    # normalize activity markov model
    activityMarkov[0][:] /= np.linalg.norm(activityMarkov[0][:], 1)
    activityMarkov[1][:] /= np.linalg.norm(activityMarkov[1][:], 1)
    
    # matrix to store calculated alpha values
    alphas = np.zeros((len(alphaTrees), alphaTrees[0].height), dtype = np.float)
        
    # normalize and store alpha values
    for i in xrange(len(alphaTrees)):
        # avoid division by zero
        if (alphaTrees[i].reuseCount[0,0] or alphaTrees[i].reuseCount[0,1]):
            alphas[i, :] = np.true_divide(alphaTrees[i].reuseCount[:, 1], \
                alphaTrees[i].reuseCount[:, 0] + alphaTrees[i].reuseCount[:, 1])
    
    """ structures stored in the profile:
    
        - blockSize: input argument value
        
        - alphaRatio: input argument value
        
        - reusePMF: probability mass function where the i-th index represents 
        the probability of reuse-distance = (i - 1) occuring. Index 0 indicates
        the probability of a not-previously-accessed block being accessed 
        (reuse-distance = Inf). This also represents a compulsory cache miss
        
        - loadProp: array where the i-th element corresponds to the probability
        of a load (read) from memory occuring for reused-distance = (i-1).
        Again, index 0 corresponds to prob(load) for a not-previously-accessed
        block access. This is used to generate ld/str info for each access
        based on the access' reuse distance
        
        - activityMarkov: markov model for the probability of an inactive/active
        memory cycle given whether the previous memory cycle was inactive/active.
        0 corresponds to inactive, and 1 corresponds to active. Thus, the value
        of activityMarkov[0][0] is the probability of an inactive cycle occuring
        given the previous cycle was inactive
        
        - alphas: matrix where the i-th row corresponds the the alpha values
        for the (alphaRatio * i) up to (alphaRation * (i+1)) reuse distances.
        These are used to iteratively project accesses to memory blocks into
        one half of the memory block based on which half (aka subset) of the
        block was accessed previously. This helps to model the spatial
        locality of the memory reference stream"""
    
    # append 'h5' file extension
    outputFile = outputFile + ".h5"
    
    # save application profile to file
    outputFile = h5.File(outputFile, 'w')
    outputFile.create_dataset('blockSize', data = blockSize, dtype = np.int)
    outputFile.create_dataset('alphaRatio', data = alphaRatio, dtype = np.float)
    outputFile.create_dataset('reusePMF', data = np.asarray(reusePMF, dtype = np.float))
    outputFile.create_dataset('loadProp', data = np.asarray(loadProp, dtype = np.float))
    outputFile.create_dataset('activityMarkov', data = activityMarkov)
    outputFile.create_dataset('alphas', data = alphas)
    outputFile.close()
    
#
## main function
#
 
if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise IndexError("Invalid number of arguments. Only config file should be specified")
            
        # setup config parser with default args
        config = ConfigParser.RawConfigParser({'alphaRatio': 0, 'blockSize': 512, 'regEx': None})
        config.read(sys.argv[1])
        
        # pull arguments
        traceFile = config.get('profiler', 'traceFile')
        outputFile = config.get('profiler', 'outputFile')
        alphaRatio = int(config.get('profiler', 'alphaRatio'))
        blockSize = int(config.get('profiler', 'blockSize'))
        regEx = config.get('profiler', 'regEx')
        
        # generate the profile
        GenerateApplicationProfile(traceFile, outputFile, alphaRatio, blockSize, regEx)
    
    except IOError as error:
        print "IOError: ", error
        
    except ValueError as error:
        tb = sys.exc_info()[2]
        traceback.print_tb(tb)
        print "ValueError: ", error
        
    except re.error as error:
        print "Invalid regEx: ", error
        
    except ConfigParser.NoOptionError as error:
        print "Invalid Args: ", error, "\n"
        print usage_info
    
    except ConfigParser.NoSectionError as error:
        print "Invalid Config: ", error, "\n"
        print usage_info
    
    except IndexError as error:
        print "IndexError: ", error, "\n"
        print usage_info
    
            
