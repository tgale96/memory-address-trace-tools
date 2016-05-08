""" filename: PreProcessing.py
    contents: this files contains the routines used by "TraceGenerator" to
    mix application profiles
    
    author: Trevor Gale
    date: 3.5.16"""

import numpy as np

# TODO: issue mixing things without alphaRatios that are multiples of each other?
def BuildAlphaForest(appProfiles, alphaForest, weights, alphaRatio):
    """ BuildAlphaForest: combines all alpha values for the input application
        profiles and builds the alphaForest accordingly
        
        args:
            - appProfiles: list of open file handles for each application profile
            - alphaForest: initialized list of AlphaTrees
            - weights: python list specifying the weights of each profile
            - alphaRatio: ratio of num_reuse_distances / alphaTree
            
        returns: alphaRatio to use for generation"""
    # get number of profiles
    numProfiles = len(appProfiles)
    numAlpha = len(alphaForest[0].reuseCount)
    
    # structures for alphaRatios and all alpha values
    allAlphaRatios = np.zeros(len(appProfiles), dtype = np.float)
    alphaValues = []

    weightSum = 0
    for i in xrange(numProfiles):
        # store alphaRatio and alpha values
        allAlphaRatios[i] = appProfiles[i]['alphaRatio'][()]
        alphaValues.append(appProfiles[i]['alphas'])
        
        # sum up all weights
        weightSum += weights[i]
    
    # for each tree in the alphaForest
    for i in xrange(len(alphaForest)):
        alphas = np.zeros(numAlpha, dtype = np.float)
        reuseDistance = i * alphaRatio


        # generate alpha values 
        for j in xrange(numProfiles):
            if allAlphaRatios[j] == float('Inf'):
                index = 0
            else:
                index = reuseDistance/allAlphaRatios[j]
            
            alphas = np.add(alphas, alphaValues[j][index] \
               * weights[j])
        
        # normalize by weightSum
        alphas /= weightSum
        
        # load into this tree
        alphaForest[i].LoadAlphas(alphas)