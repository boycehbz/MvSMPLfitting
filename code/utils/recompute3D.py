import numpy as np
from utils.utils import get_rot_trans

def nomalized(z):
    norm = np.linalg.norm(z)
    z = z / norm
    return z

def fill_nMat(n):
    nMat = np.dot(n.reshape(3,1), n.reshape(1,3))
    nMat = np.eye(3) - nMat
    return nMat

def recompute3D(extris, intris, keypoints):

    assert len(extris) == len(intris) and len(extris) == len(keypoints)

    # homegeneous
    keypoints = np.concatenate(keypoints, axis=0)
    joint_conf = keypoints[:,:,2].copy()
    keypoints[:,:,2] = 1.
    # (keypoints, 2, 1, axis=2)

    num_joint = keypoints.shape[1]

    AtA = np.zeros((num_joint,3,3))
    Atb = np.zeros((num_joint,1,3))
    skelPos = np.zeros((3,num_joint))

    ts, Rs = get_rot_trans(extris)

    for v in range(len(extris)):
        conf = joint_conf[v]
        intri = np.linalg.inv(intris[v])
        R = Rs[v]
        t = ts[v]
        keps = keypoints[v]
        for i in range(len(keps)):
            n = np.dot(intri, keps[i])
            n = nomalized(n)
            nMat = fill_nMat(n)
            nMat = np.dot(R.T, nMat)
            AtA[i] += np.dot(nMat, R) * conf[i]
            Atb[i] += np.dot(-nMat, t) * conf[i]
    
    AtA = AtA.astype(np.float32)
    for i in range(len(keps)):
        # l, d = LDLT(AtA[i])
        # y = np.linalg.solve(l, Atb[i].T)
        # skelPos[:,i] = np.linalg.solve(d, y).reshape(3,)
        skelPos[:,i] = np.linalg.solve(AtA[i], Atb[i].T).reshape(3,)
        #skelPos.col(i) = AtA[i].ldlt().solve(Atb[i])
    skelPos = skelPos.T
    return skelPos