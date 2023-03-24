import numpy as np
import pytest
import nibabel as nb
from nifti import voxel_sizes, dir_cosines


@pytest.mark.parametrize(
    "input_matrix,expected_sizes",
    [
        (np.diag([1.0, 2.0, 3.0, 1.0]), [1.0, 2.0, 3.0]),
        (np.diag([2.0, -1.0, 3.0, 1.0]), [2.0, 1.0, 3.0]),
        (
            np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            [2.0, 1.0, 3.0],
        ),
    ],
)
def test_voxels(input_matrix, expected_sizes):
    assert np.allclose(voxel_sizes(input_matrix), expected_sizes)


@pytest.mark.parametrize(
    "expected_sizes",
    [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]],
)
@pytest.mark.parametrize("testnumber", list(range(10)))
@pytest.mark.parametrize("flip_y", [True, False])
def test_fuzzing(expected_sizes, testnumber, flip_y):
    rotmat = nb.eulerangles.euler2mat(
        z=np.random.uniform(-np.pi / 4, np.pi / 4),
        y=np.random.uniform(-np.pi / 4, np.pi / 4) + np.pi * flip_y,
        x=np.random.uniform(-np.pi / 4, np.pi / 4),
    )
    input_matrix = np.eye(4)
    input_matrix[:-1, :-1] = rotmat * expected_sizes
    assert np.allclose(voxel_sizes(input_matrix), expected_sizes)


@pytest.mark.parametrize(
    "xform,xform_dir",
    [
        (np.diag([1.0, 2.0, 3.0, 1.0]), np.eye(3)),
        (np.diag([2.0, -1.0, 3.0, 1.0]), np.diag([1.0, -1.0, 1.0])),
    ],
)
def test_dir_cosines(xform, xform_dir):
    assert np.allclose(dir_cosines(xform), xform_dir)
