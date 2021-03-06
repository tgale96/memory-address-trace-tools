""" filename: AlphaTree
    contents: This file contains the AlphaTree class as well as the definition
    of all member function. This class is used to collect reuse/non-reuse stats
    for all subsets of a given memory block size
    
    author: Trevor Gale
    date: 4.15.16"""

from math import log
import numpy as np

class AlphaTree:
    """ class alphaTree: tree where each edge represents a subset of some
        superset block of memory. Used to track reuse of subsets within 
        supersets & calcualte probabilities of subset reuse for each level
        ("alpha values"). Stores alpha values down to the 4-byte word"""
        
    def __init__(self, rootSize = 512, bins = 3):
        """ __init__: initializes alphaTree data structures for edges as well
            as global reuse and non-reuse counters for each level in the tree
            
            args: 
                - rootSize: blocksize of biggest superset in the tree i.e the 
                blocksize of the block represented by the root. Default is 
                512 bytes
                - bins: number of bins to divide reuse distances into
                
            NOTE: reuseCount is stored so that each index corresponds to the
            reuse at that HEIGHT in the tree i.e. index 6 is used to calculate
            \alpha(256, 512) and index 0 is used to calculate \alpha(4, 8)
            (assuming default height setting)"""            
        # validate input
        if rootSize % 2 or rootSize < 8:
            raise ValueError("(in AlphaTree.__init__) rootSize must be power of 2 >= 8")
            
        # save rootSize
        self.rootSize = rootSize
        
        # save number of bins
        self.bins = bins
        
         # height of the tree
        self.height = int(log(rootSize / 4, 2))

        # tree structure
        self.tree = np.zeros(rootSize / 2 - 1, dtype = np.bool)
        
        # reuse counters for each level
        self.reuseCount = np.zeros((self.bins, self.height, 2), dtype = np.float) 
    
    def GetChild(self, parent, side):
        """ GetChild: returns the identifier of the desired child in tree
        
            args:
                - parent: identifier of parent node
                - side: 0 for left child, 1 for right child"""
        
        if side: # if right
            return (parent * 2) + 2
        return (parent * 2) + 1
        
    def GetLeftChild(self, parent):
        """ GetLeftChild: returns identifier of left child in tree
        
            args:
                - parent: identifier of parent node"""
        return (parent * 2) + 1
    
    def GetRightChild(self, parent):
        """ GetRightChild: returns identifier of right child in tree
        
            args:
                - parent: identifier of parent node"""
        return (parent * 2) + 2
    
    def GetParent(self, child):
        """ GetParent: returns identifier of parent in tree for input child
        
            args: 
                - child: identifier of child node"""
        return int(child - 1) / 2
    
    def GetSibling(self, child):
        """ GetSibling: returns identifier of the sibling of this child in tree
        
            args:
                - child: identifier of child node"""
        parent = self.GetParent(child) # get parent id
        
        return self.GetChild(parent, child % 2)
    
    def GetTreeLevel(self, blockSize):
        """ GetTreeLevel: converts input blockSize into the height of that 
            block in the tree
            
            args:
                - blockSize: size of block to find correponding height to"""
        return int(log(blockSize/4, 2) - 1)

    def ProcessAccess(self, memAddress, reuseDist):
        """ ProcessAccess: handle a memory access by iterating through the
            tree while marking used edges and recording reuse of non-reuse
            at each level
            
            args:
                - memAddress: address at which the memory access occurred
                - reuseDist: reuse distance at which access occured"""
        # set bin index
        if reuseDist > (self.bins - 1):
            reuseDist = self.bins - 1
            
        self.updateTree(memAddress, 0, self.rootSize, reuseDist) # recursively update tree
        
    def updateTree(self, memAddress, nodeID, blockSize, reuseDist):
        """ updateTree: recursively updates appropriate edges in the 
            alphaTree
            
            args:
                - memAddress: address at which the memroy access occurred
                - nodeID: ID of the current node
                - blockSize: subset size represented by this level in the tree
                - reuseDist: reuse distance bin in which access occured"""
                
        # if we reached end of tree        
        if blockSize == 4:
            return
        
        # get index of used subset and it's sibling in tree
        subsetID = ((blockSize >> 1) & memAddress) >> int(log(blockSize >> 1, 2))
        usedSubset = self.GetChild(nodeID, subsetID)
        sibling = self.GetSibling(usedSubset)
        
        # get htis block's height in the tree
        height = self.GetTreeLevel(blockSize)
        
        # handle reuse of this subset
        if not (self.tree[usedSubset] or self.tree[sibling]):
            # mark used edge
            self.tree[usedSubset] = True
            
        elif self.tree[usedSubset]:
            # indicate reuse
            self.reuseCount[reuseDist][height][1] += 1
            
        else:
            # indicate non-reuse
            self.reuseCount[reuseDist][height][0] += 1
            
            # record which subset was used
            self.tree[usedSubset] = True
            self.tree[sibling] = False

        # handle next block        
        self.updateTree(memAddress, usedSubset, blockSize >> 1, reuseDist)
        
    def LoadAlphas(self, alphaValues):
        """ LoadAlphas: loads array of alpha values into self.alphas 
        
            args: 
                - alphaValues: np array self.bins x self.height array where the i-th 
                row corresponds to reuse distances of i (i = 2 for rd >= 2),
                and each row contains the series of alpha values to load into
                self.reuseCount"""
        # validate input dimensions
        if alphaValues.shape != (self.bins, self.height, 2):
            raise ValueError("(in AlphaTree.LoadAlpha) input matrix must be num_bins x tree_height x 2")

        # store input values        
        self.reuseCount = alphaValues
        
    def GenerateAccess(self, reuseDist):
        """ GenerateAccess: selects 4-byte word to access based on the previous
            accesses and the alpha values stored in self.alphas
            
            args: 
                - reuseDist: reuse distance at which access occured
            
            return: bottom N bits to append to block address"""
        # set bin index
        if reuseDist > (self.bins - 1):
            reuseDist = self.bins - 1
            
        return self.SelectWord(0, self.rootSize, reuseDist)
    
    def SelectWord(self, nodeID, blockSize, reuseDist):
        """ SelectWord: iterates through tree to select 4-byte word to access
        
            args:
                - nodeID: ID of the current node
                - blockSize: subset size represented by this tree level
                - reuseDist: reuse distance bin in which the access occurred"""
        # if we reached end of tree        
        if blockSize == 4:
            return 0
        
        # get left and right children
        leftChild = self.GetChild(nodeID, 0)
        rightChild = self.GetChild(nodeID, 1)
                
        # get this blocks height in the tree
        height = self.GetTreeLevel(blockSize)
        
        # if tree not previously accessed
        if not(self.tree[leftChild] or self.tree[rightChild]):
            # select from uniform distribution
            subsetIndex = np.random.choice(2)
            usedSubset = self.GetChild(nodeID, subsetIndex)
            
            # mark this subset as used
            self.tree[usedSubset] = True
            
            # return appropriate bit
            return self.SelectWord(usedSubset, blockSize >> 1, reuseDist) \
                | (subsetIndex * (blockSize >> 1))
                
        # else, whether to reuse subset or not
        reuse = np.random.choice(2, p = self.reuseCount[reuseDist, height,:])
        
        # identify previously accessed child
        prevSubset = leftChild
        prevIndex = 0
        if self.tree[rightChild]:
            prevSubset = rightChild 
            prevIndex = 1
            
        # update usage if necessary
        if not reuse:
            usedSubset = self.GetSibling(prevSubset)
            
            self.tree[usedSubset] = True
            self.tree[prevSubset] = False
            subsetIndex = (prevIndex+1) % 2
        else:
            subsetIndex = prevIndex
            usedSubset = prevSubset

        # return appropriate bit
        return self.SelectWord(usedSubset, blockSize >> 1, reuseDist) \
            | (subsetIndex * (blockSize >> 1))

    def NormalizeReuseCount(self):
        """ NormalizeReuseCount: normalizes each row of alpha values so 
            that that represent probabilities"""
        # for each bin
        for i in xrange(self.bins):
            for j in xrange(self.height):
                norm = np.linalg.norm(self.reuseCount[i, j, :], 1)
                
                if not norm:
                    # set reuse to 1.0
                    self.reuseCount[i, j, 1] = 1.0
                    continue
                
                # normalize values
                self.reuseCount[i, j, :] /= norm
            
    
        
        
        