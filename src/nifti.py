import numpy as np
import nibabel as nb


def voxel_sizes(xform):
    """Calculates the voxel sizes given the s/q-form matrix of a NIfTI file."""
    return np.linalg.norm(xform[:-1, :-1], axis=0)


def dir_cosines(xform):
    """Calculates the orientation of the s/q-form matrix of a NIfTI file."""
    return xform[:-1, :-1] / voxel_sizes(xform)


