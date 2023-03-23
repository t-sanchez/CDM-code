import numpy as np


def voxel_sizes(xform):
    """Calculates the voxel sizes given the s/q-form matrix of a NIfTI file."""
    return np.linalg.norm(xform[:-1, :-1], axis=0)
