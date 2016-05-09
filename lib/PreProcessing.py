""" filename: PreProcessing.py
    contents: this files contains the routines used by "TraceGenerator" to
    mix application profiles
    
    author: Trevor Gale
    date: 3.5.16"""

import numpy as np
from AlphaTree import AlphaTree

def BuildAlphaForest(appProfiles, weights, alphaForest, wsSize, blockSize):
    """ BuildAlphaForest: combines all alpha values for the input application
        profiles and builds the alphaForest accordingly
        
        args:
            - appProfiles: list of open file handles for each application profile
            - weights: python list specifying the weights of each profile
            - alphaForest: list to store alphaTrees in
            - wsSize: max number of cache blocks accessed by any"""
    # get number of profiles
    numProfiles = len(appProfiles)
    
    # get shape of first alpha matrix
    alphaShape = np.asarray(appProfiles[0]['alphas']).shape
    
    # matrix to store alpha values
    alphaValues = np.zeros((wsSize, alphaShape[1], alphaShape[2], 2), dtype = np.float)

    # create linear combination of all profiles alpha values
    for i in xrange(numProfiles):
        temp = np.asarray(appProfiles[i]['alphas'])
        
        # resize alpha matrix
        temp.resize(alphaValues.shape)
            
        # add to sum
        alphaValues += temp * weights[i]    
        
    # for each cache block
    for i in xrange(wsSize):        
        # create an AlphaTree
        alphaForest.append(AlphaTree(blockSize, alphaShape[1]))
        
        # load alpha values
        alphaForest[i].LoadAlphas(alphaValues[i])
        
        # normalize alpha values
        alphaForest[i].NormalizeReuseCount()

def BuildMarkovModel(appProfiles, weights, activityMarkov):
    """ BuildMarkovModel: creates linear combination of all the markov 
        activity models in the input profiles
        
        args:
            - appProfiles: list of open file handles for each application profile
            - weights: list of weights for each profile
            - activityMarkov: 2x2 markov matrix to store results in"""
    numProfiles = len(appProfiles)
    
    # for each profile
    for i in xrange(numProfiles):        
        # create weighted sum of activity markov models
        activityMarkov += weights[i] * np.asarray(appProfiles[i]['activityMarkov'])
    
    # normalize markov model
    activityMarkov[0][:] /= np.linalg.norm(activityMarkov[0][:], 1)
    activityMarkov[1][:] /= np.linalg.norm(activityMarkov[1][:], 1)

def BuildReusePMF(appProfiles, weights, reusePMF):
    """ BuildReusePMF: creates a linear combination of the reusePMF's of all
        the input profiles 
        
        args:
            - appProfiles: list of open file handles for each application profile
            - weights: list of weights for each profile
            - reusePMF: numpy array to store result in"""
    numProfiles = len(appProfiles)
    numReuseDistances = len(reusePMF)
    
    # for each profile
    for i in xrange(numProfiles):
        # weighted sum of reusePMFs
        temp = np.asarray(appProfiles[i]['reusePMF'])
        temp.resize(numReuseDistances)
        reusePMF += temp * weights[i]
        
    # normalize reusePMF
    reusePMF /= np.linalg.norm(reusePMF, 1)

def BuildLoadProp(appProfiles, weights, loadProp):
    """ BuildLoadProp: creates a linear combination of loadProp's of all
        the input profiles
        
        args: 
            - appProfiles: list of open file handles for each application profile
            - weights: list of weights for each profile
            - loadProp: numpy array to store reults in"""
    numProfiles = len(appProfiles)
    numReuseDistances = len(loadProp)

    # for each profile
    for i in xrange(numProfiles):        
        # weights sume of load proportions
        temp = np.asarray(appProfiles[i]['loadProp'])
        temp.resize(numReuseDistances)
        loadProp += temp * weights[i]
    
    # normalize loadProp
    weightSum = 0
    for i in xrange(numProfiles):
        weightSum += weights[i]
    loadProp /= weightSum

def BuildWorkingSet(appProfiles):
    """ BuildWorkingSet: selects the largest working set out of all the 
        input application profiles
        
        args:
            - appProfiles: list of open file handles for each application profile"""
    numProfiles = len(appProfiles)
    wsSize = 0
    
    for i in xrange(numProfiles):
        if len(appProfiles[i]['workingSet']) > wsSize:
            ws = i
    
    return np.asarray(appProfiles[ws]['workingSet'])
