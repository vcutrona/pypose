# code adapted from https://gist.github.com/pweids/8f1b041ed4c308c40c8dda5df0721f31

from math import cos, sin, sqrt, pi, acos

import numpy as np
import numpy.typing as npt


class AxisAngle:
    def __init__(self,
                 x: float = None,
                 y: float = None,
                 z: float = None,
                 theta: float = None,
                 rot_vec: npt.NDArray = None,
                 rot_mat: npt.NDArray = None):
        if x and y and z and theta:
            self._x = x
            self._y = y
            self._z = z
            self._theta = theta
        if rot_vec is not None:
            self._rot_vec = rot_vec
            self._compute_components_from_rot_vec()
        elif rot_mat is not None:
            self._rot_mat = rot_mat
            self._compute_components_from_rot_mat()

    def get_rotation_vector(self) -> npt.NDArray:
        if self._rot_vec is None:
            self._rot_vec = np.array([self._angle * self._x, self._angle * self._y, self._angle * self._z])
        return self._rot_vec

    def get_rotation_matrix(self) -> npt.NDArray:
        if self._rot_mat is not None:
            return self._rot_mat
        if self._angle == 0:
            return np.eye(3)

        c = cos(self._angle)
        s = sin(self._angle)
        t = 1.0 - c

        m00 = c + self._x * self._x * t
        m11 = c + self._y * self._y * t
        m22 = c + self._z * self._z * t

        tmp1 = self._x * self._y * t
        tmp2 = self._z * s

        m10 = tmp1 + tmp2
        m01 = tmp1 - tmp2

        tmp1 = self._x * self._z * t
        tmp2 = self._y * s

        m20 = tmp1 - tmp2
        m02 = tmp1 + tmp2

        tmp1 = self._y * self._z * t
        tmp2 = self._x * s

        m21 = tmp1 + tmp2
        m12 = tmp1 - tmp2

        return np.array([
            [m00, m01, m02],
            [m10, m11, m12],
            [m20, m21, m22]
        ])

    def _compute_components_from_rot_vec(self):
        self._rot_mat = None  # we do this to reset the lazy calculation from getRotationMatrix

        self._angle = sqrt(pow(self._rot_vec[0], 2) + pow(self._rot_vec[1], 2) + pow(self._rot_vec[2], 2))
        if self._angle == 0:
            self._y = 0
            self._z = 0
            self._x = 1
        else:
            self._x = self._rot_vec[0] / self._angle
            self._y = self._rot_vec[1] / self._angle
            self._z = self._rot_vec[2] / self._angle

    def _compute_components_from_rot_mat(self):
        self._rot_vec = None  # we do this to reset the lazy calculation from getRotVec
        epsilon1 = 0.01  # margin to allow for rounding errors
        epsilon2 = 0.1  # margin to distinguish between 0 and 180 degrees

        # optional check that input is pure rotation, 'isRotationMatrix' is defined at:
        # https://www.euclideanspace.com/maths/algebra/matrix/orthogonal/rotation/
        if abs(self._rot_mat[0, 1] - self._rot_mat[1, 0]) < epsilon1 and \
                abs(self._rot_mat[0, 2] - self._rot_mat[2, 0]) < epsilon1 and \
                abs(self._rot_mat[1, 2] - self._rot_mat[2, 1]) < epsilon1:
            # singularity found
            # first check for identity matrix which must have +1 for all terms  in leading diagonal
            # and zero in other terms
            if abs(self._rot_mat[0, 1] + self._rot_mat[1, 0]) < epsilon2 and \
                    abs(self._rot_mat[0, 2] + self._rot_mat[2, 0]) < epsilon2 and \
                    abs(self._rot_mat[1, 2] + self._rot_mat[2, 1]) < epsilon2 and \
                    abs(self._rot_mat[0, 0] + self._rot_mat[1, 1] + self._rot_mat[2, 2] - 3) < epsilon2:
                # this singularity is identity matrix so angle = 0
                self._angle = 0
                self._y = 0
                self._z = 0
                self._x = 1
                return

            # otherwise, this singularity is angle = 180
            angle = pi
            xx = (self._rot_mat[0, 0] + 1) / 2
            yy = (self._rot_mat[1, 1] + 1) / 2
            zz = (self._rot_mat[2, 2] + 1) / 2
            xy = (self._rot_mat[0, 1] + self._rot_mat[1, 0]) / 4
            xz = (self._rot_mat[0, 2] + self._rot_mat[2, 0]) / 4
            yz = (self._rot_mat[1, 2] + self._rot_mat[2, 1]) / 4
            if (xx > yy) and (xx > zz):  # m[0][0] is the largest diagonal term
                if xx < epsilon1:
                    x = 0
                    y = 0.7071
                    z = 0.7071
                else:
                    x = sqrt(xx)
                    y = xy / x
                    z = xz / x

            elif yy > zz:  # m[1][1] is the largest diagonal term
                if yy < epsilon1:
                    x = 0.7071
                    y = 0
                    z = 0.7071
                else:
                    y = sqrt(yy)
                    x = xy / y
                    z = yz / y
            else:  # m[2][2] is the largest diagonal term so base result on this
                if zz < epsilon1:
                    x = 0.7071
                    y = 0.7071
                    z = 0
                else:
                    z = sqrt(zz)
                    x = xz / z
                    y = yz / z

            self._x = x
            self._y = y
            self._z = z
            self._angle = angle
            return  # return 180 deg rotation

        # as we have reached here there are no singularities, so we can handle normally
        s = sqrt((self._rot_mat[2, 1] - self._rot_mat[1, 2]) * (self._rot_mat[2, 1] - self._rot_mat[1, 2])
                 + (self._rot_mat[0, 2] - self._rot_mat[2, 0]) * (self._rot_mat[0, 2] - self._rot_mat[2, 0])
                 + (self._rot_mat[1, 0] - self._rot_mat[0, 1]) * (
                         self._rot_mat[1, 0] - self._rot_mat[0, 1]))  # used to normalise
        if abs(s) < 0.001:
            s = 1
        # prevent divide by zero, should not happen if matrix is orthogonal and should be
        # caught by singularity test above, but I've left it in just in case
        angle = acos((self._rot_mat[0, 0] + self._rot_mat[1, 1] + self._rot_mat[2, 2] - 1) / 2)
        x = (self._rot_mat[2, 1] - self._rot_mat[1, 2]) / s
        y = (self._rot_mat[0, 2] - self._rot_mat[2, 0]) / s
        z = (self._rot_mat[1, 0] - self._rot_mat[0, 1]) / s
        self._x = x
        self._y = y
        self._z = z
        self._angle = angle
