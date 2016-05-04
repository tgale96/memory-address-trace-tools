import pytest
import numpy as np
from lib.AlphaTree import AlphaTree

def test_process_access():
    """ tests AlphaTree::ProcessAccess function with sequence of memory accesses"""
    # init default AlphaTree
    a = AlphaTree()
    
    memAddr = 0b111111111
    
    a.ProcessAccess(memAddr)
    
    # count should be all 0's
    assert np.array_equal(a.reuseCount, np.zeros((7, 2), dtype = np.float))
    
    a.ProcessAccess(memAddr)
    
    sol = np.zeros((7, 2), dtype = np.float)
    sol[:, 1] = 1.0
    
    # should have every level reused
    assert np.array_equal(a.reuseCount, sol)
    
    memAddr = 0b111110000
    
    a.ProcessAccess(memAddr)
    
    # top 5 reused, then 1 non-reuse & 1 nothing
    sol[2:7, 1] += 1
    sol[1, 0] += 1

    assert np.array_equal(a.reuseCount, sol)
    
    memAddr = 0b000000000

    a.ProcessAccess(memAddr)

    # should change except for first layer non-reuse
    sol[6, 0] += 1    

    assert np.array_equal(a.reuseCount, sol)
    
    memAddr = 0b000000100
    
    a.ProcessAccess(memAddr)
    
    # 6-1 reuse and 0 non-reuse
    sol[1:7, 1] += 1
    sol[0, 0] += 1
    
    assert np.array_equal(a.reuseCount, sol)
    
    memAddr = 0b111000000
    
    a.ProcessAccess(memAddr)
    
    # 6th non-reuse, 5-4 reuse, 3rd non-reuse, 2-0 nothing
    sol[6, 0] += 1
    sol[4:6, 1] += 1
    sol[3, 0] += 1
    
    assert np.array_equal(a.reuseCount, sol)
    
def test_get_family():
    """ tests node family functions for AlphaTree"""
    # init AlphaTree    
    a = AlphaTree()
    
    # start with root   
    nodeID = 0
    
    # check left child ID
    assert 1 == a.GetChild(nodeID, 0)
    
    # check right child ID
    assert 2 == a.GetChild(nodeID, 1)
    
    nodeID = 1
    
    # check parent is root
    assert 0 == a.GetParent(nodeID)
    
    # check sibling is 2
    assert 2 == a.GetSibling(nodeID)
    
    # check left child
    assert 3 == a.GetChild(nodeID, 0)
    
    # check right child
    assert 4 == a.GetChild(nodeID, 1)
    
    nodeID = 3
    
    # check left child
    assert 7 == a.GetChild(nodeID, 0)
    
    # check right child
    assert 8 == a.GetChild(nodeID, 1)
    
    # check parent
    assert 1 == a.GetParent(nodeID)
    
    # check sibling is 4
    assert 4 == a.GetSibling(nodeID)
    
def test_tree_level():
    """ tests AlphaTree::GetTreeLevel function"""
    # init AlphaTree()
    a = AlphaTree()
    
    block = 512
    
    # check level is top level
    assert 6 == a.GetTreeLevel(block)

    block = 8

    assert 0 == a.GetTreeLevel(block)

    block = 32

    assert 2 == a.GetTreeLevel(block)

def test_larger_root_size():
    """ tests AlphaTree::ProcessAccess with rootSize = 1024"""
    # init AlphaTree
    a = AlphaTree(1024)
    
    memAddr = 0b1111111111
    
    a.ProcessAccess(memAddr)
    
    sol = np.zeros((8,2), dtype = np.float)
    
    assert np.array_equal(a.reuseCount, sol)
    
    a.ProcessAccess(memAddr)
    
    # all reused
    sol[:, 1] += 1
    
    assert np.array_equal(a.reuseCount, sol)
    
    memAddr = 0b0000000000
    
    a.ProcessAccess(memAddr)
    
    # one non-reuse at top
    sol[7, 0] += 1
    
    assert np.array_equal(a.reuseCount, sol)
    
    memAddr = 0b1111100000
    
    a.ProcessAccess(memAddr)
    
    # 7 non-reuse, 3-6 reuse, 2 non-reuse, else nothing
    sol[7, 0] += 1
    sol[3:7, 1] += 1
    sol[2, 0] += 1
    
    assert np.array_equal(a.reuseCount, sol)
    
def test_smaller_root_size():
    """ tests AlphaTree::ProcessAccess with rootSize = 256"""
    # init AlphaTree
    a = AlphaTree(256)
    
    memAddr = 0b11111111
    
    a.ProcessAccess(memAddr)
    
    sol = np.zeros((6, 2), dtype = np.float)
    
    assert np.array_equal(a.reuseCount, sol)
    
    memAddr = 0b00000000

    a.ProcessAccess(memAddr)
    
    sol[5, 0] += 1
    
    assert np.array_equal(a.reuseCount, sol)    
    
    memAddr = 0b11110000
    
    a.ProcessAccess(memAddr)
    
    # 5 non-reuse, 2-4 reuse, 1 non-reuse
    sol[5, 0] += 1
    sol[1,0] += 1
    sol[2:5, 1] += 1
    
    assert np.array_equal(a.reuseCount, sol)

def test_load_alphas():
    """ tests AlphaTree::LoadAlphas to ensure values are stored 
        correctly"""
    # init AlphaTree
    a = AlphaTree()
    
    test_alphas = np.arange(7, dtype = np.float)
    test_alphas /= 7.0
    
    a.LoadAlphas(test_alphas)
    
    assert np.array_equal(test_alphas, a.reuseCount[:,1])
    assert np.array_equal(1-test_alphas, a.reuseCount[:,0])

def test_generate_access():
    """ test AlphaTree::GenerateAccess to ensure accesses are tracked
        correctly and ouput is generated correctly based on the path
        take when traversing the tree from root to leaf"""
    # init AlphaTree
    a = AlphaTree()
    
    # load alpha values
    test_alphas = np.arange(7, dtype = np.float)
    test_alphas /= 7.0
    
    a.LoadAlphas(test_alphas)
    
    # test first access
    test = a.GenerateAccess()
    
    # traverse tree to find theoretical output
    node = 0
    block = 512
    results = 0
    while a.GetRightChild(node) < len(a.tree):
        if a.tree[a.GetLeftChild(node)]:
            node = a.GetLeftChild(node)
        else:
            node = a.GetRightChild(node)
            results = results | (block >> 1)
        block = block >> 1
    
    # compare results
    assert test == results
    
    # test second access
    test = a.GenerateAccess()
    
    # traverse tree to find theoretical output
    node = 0
    block = 512
    results = 0
    while a.GetRightChild(node) < len(a.tree):
        if a.tree[a.GetLeftChild(node)]:
            node = a.GetLeftChild(node)
        else:
            node = a.GetRightChild(node)
            results = results | (block >> 1)
        block = block >> 1

    # compare results
    assert test == results
    
    # load different distribution
    test_alphas = np.ones(7, dtype = np.float)
    test_alphas /= 2
    
    a.LoadAlphas(test_alphas)
    
    # test 3rd access
    test = a.GenerateAccess()
    
    # traverse tree to find theoretical output
    node = 0
    block = 512
    results = 0
    while a.GetRightChild(node) < len(a.tree):
        if a.tree[a.GetLeftChild(node)]:
            node = a.GetLeftChild(node)
        else:
            node = a.GetRightChild(node)
            results = results | (block >> 1)
        block = block >> 1

    # compare results
    assert test == results

    
    
    
    
    
    
    
    
    
    
    