import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v=np.asarray(v)
    w=np.asarray(w)

    dp=np.dot(v,w)

    mag_v=np.linalg.norm(v)
    mag_w=np.linalg.norm(w)

    if mag_v<10**(-10) or mag_w<10**(-10):
        return np.nan

    cos=dp/(mag_v*mag_w)
    cos=np.clip(cos,-1,1)

    return np.arccos(cos)