from transforms3d.quaternions import qinverse, rotate_vector, qmult


def mapfree_convert_world2cam_to_cam2world(q, t):
    qinv = qinverse(q)
    tinv = -rotate_vector(t, qinv)
    return qinv, tinv