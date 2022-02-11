# code adapted from https://gist.github.com/pweids/dbbf1903b0d4cb857ddaa42725b61c
from typing import Union, List

import numpy as np
import numpy.typing as npt

from pypose.axis_angle import AxisAngle
from pypose.matrix import Matrix


class Pose:

    def __init__(self, pose: Union[List, npt.NDArray] = None, matrix: Union[List[List[float]], npt.NDArray] = None):
        if pose is not None:
            self._pose = np.array(pose) if type(pose) == list else pose
        elif matrix is not None:
            rot = AxisAngle(rot_mat=np.array([
                [matrix[0][0], matrix[0][1], matrix[0][2]],
                [matrix[1][0], matrix[1][1], matrix[1][2]],
                [matrix[2][0], matrix[2][1], matrix[2][2]]
            ]))
            rv = rot.get_rotation_vector()
            self._pose = np.array([matrix[0][3], matrix[1][3], matrix[2][3], rv[0], rv[1], rv[2]])

    def get_rotation_vector(self) -> npt.NDArray:
        return self._pose[3:6]

    def get_translation_vector(self) -> npt.NDArray:
        return self._pose[0:3]

    def invert(self) -> 'Pose':
        rot_mat = self.get_transform_matrix()
        inv_mat = rot_mat.invert_transformation()
        angle = AxisAngle(rot_mat=inv_mat.sub_matrix(0, 3, 0, 3).to_array())
        rot_vec = angle.get_rotation_vector()
        trans_mat = inv_mat.sub_matrix(0, 3, 3, 4).to_array()

        return Pose(pose=[trans_mat[0, 0], trans_mat[1, 0], trans_mat[2, 0], rot_vec[0], rot_vec[1], rot_vec[2]])

    def get_transform_matrix(self) -> 'Matrix':
        trans = self.get_translation_vector()
        rot = self.get_rotation_vector()

        angle = AxisAngle(rot_vec=rot)
        rot_mat = angle.get_rotation_matrix()

        return Matrix(matrix=np.array([
            [rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], trans[0]],
            [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], trans[1]],
            [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], trans[2]],
            [0, 0, 0, 1]
        ]))

    def trans(self, p2: 'Pose') -> 'Pose':
        return Pose(matrix=self.get_transform_matrix().multiply(p2.get_transform_matrix()).to_array())

    def to_array(self) -> npt.NDArray:
        return self._pose

    def __eq__(self, other) -> bool:
        return isinstance(other, Pose) and np.allclose(self._pose, other._pose, rtol=1.e-13, atol=1.e-13)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return self._pose.__repr__()

    def __add__(self, other) -> npt.NDArray:
        return self._pose + other.to_array()

    def __sub__(self, other) -> npt.NDArray:
        return self._pose - other.to_array()
