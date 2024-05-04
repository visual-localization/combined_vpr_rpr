from transforms3d.quaternions import qinverse, rotate_vector, qmult
import numpy as np

from typing import List

VARIANTS_ANGLE_SIN = 'sin'
VARIANTS_ANGLE_COS = 'cos'

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
    return np.real(eigenVectors[:,0])




def quat_angle_error(label, pred, variant=VARIANTS_ANGLE_SIN) -> np.ndarray:
    assert label.shape == (4,)
    assert pred.shape == (4,)
    assert variant in (VARIANTS_ANGLE_SIN, VARIANTS_ANGLE_COS), \
        f"Need variant to be in ({VARIANTS_ANGLE_SIN}, {VARIANTS_ANGLE_COS})"

    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(label.shape) != 2 or label.shape[0] != 1 or label.shape[1] != 4:
        raise RuntimeError(f"Unexpected shape of label: {label.shape}, expected: (1, 4)")

    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    if len(pred.shape) != 2 or pred.shape[0] != 1 or pred.shape[1] != 4:
        raise RuntimeError(f"Unexpected shape of pred: {pred.shape}, expected: (1, 4)")

    label = label.astype(np.float64)
    pred = pred.astype(np.float64)

    q1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    if variant == VARIANTS_ANGLE_COS:
        d = np.abs(np.sum(np.multiply(q1, q2), axis=1, keepdims=True))
        d = np.clip(d, a_min=-1, a_max=1)
        angle = 2. * np.degrees(np.arccos(d))
    elif variant == VARIANTS_ANGLE_SIN:
        if q1.shape[0] != 1 or q2.shape[0] != 1:
            raise NotImplementedError(f"Multiple angles is todo")
        # https://www.researchgate.net/post/How_do_I_calculate_the_smallest_angle_between_two_quaternions/5d6ed4a84f3a3e1ed3656616/citation/download
        sine = qmult(q1[0], qinverse(q2[0]))  # note: takes first element in 2D array
        # 114.59 = 2. * 180. / pi
        sin = np.linalg.norm(sine[1:], keepdims=True)
        sin = min(sin,1) if sin>=0 else max(sin,-1)
        angle = np.arcsin(sin) * 114.59155902616465
        angle = np.expand_dims(angle, axis=0)

    return angle.astype(np.float64)