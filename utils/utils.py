from transforms3d.quaternions import qinverse, rotate_vector, qmult
import numpy as np

from typing import List

def convert_world2cam_to_cam2world(q, t):
    qinv = qinverse(q)
    tinv = -rotate_vector(t, qinv)
    return qinv, tinv

def weightedAverageQuaternions(Q:np.ndarray, w:List[float]):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))
    weightSum = 0

    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A
        weightSum += w[i]

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)