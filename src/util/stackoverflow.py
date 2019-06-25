#coding:utf-8
import numpy as np

def find_best_trade_off(arr):
    '''
    FOR DETAILS PLEASE REFER TO
    https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    '''
    nPoints = len(arr)
    allCoord = np.vstack((range(nPoints), arr)).T
    np.array([range(nPoints), arr])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint