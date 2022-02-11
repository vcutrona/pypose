# code adapted from https://gist.github.com/pweids/5421694c969bad4bacb31599d6a20861
from typing import Union, List

import numpy as np
import numpy.typing as npt


class Matrix:

    def __init__(self, matrix: Union[List[List[float]], npt.NDArray]):
        self._matrix = np.array(matrix) if type(matrix) == list else matrix

    def to_array(self) -> npt.NDArray:
        return self._matrix

    def determinant(self) -> float:
        return np.linalg.det(self._matrix)

    def sub_matrix(self, r_start, r_end, c_start, c_end) -> 'Matrix':
        return Matrix(matrix=self._matrix[r_start:r_end, c_start:c_end])

    def inverse(self) -> 'Matrix':
        return Matrix(matrix=np.linalg.inv(self._matrix))

    def minor(self, row: int, column: int) -> 'Matrix':
        minor = np.zeros((len(self._matrix) - 1, len(self._matrix) - 1))

        for i in range(len(self._matrix)):
            for j in range(len(self._matrix[i])):
                if i == row:
                    break
                if j != column:
                    minor[i if i < row else i - 1][j if j < column else j - 1] = self._matrix[i][j]
        return Matrix(matrix=minor)

    def invert_transformation(self) -> 'Matrix':
        """
        The formula looks like this:
        |   R   | T |    |  R^-1 | -(R^-1)*T |
        |-------+---| -> |-------+-----------|
        | 0 0 0 | 1 |    | 0 0 0 |     1     |
        """
        assert len(self._matrix[0]) == len(self._matrix) and len(self._matrix) == 4
        r_matrix = self.sub_matrix(0, 3, 0, 3)
        t_matrix = self.sub_matrix(0, 3, 3, 4)

        # calculate inverse of the rotation matrix
        r_inv = r_matrix.inverse()
        # calculate - (R ^ -1) * T
        t_vector = r_inv.multiply(t_matrix).negate()

        # being explicit to make sure nothing is messed up
        return Matrix(matrix=np.append(np.append(r_inv.to_array(),
                                                 t_vector.to_array()[:, 0].reshape(3, 1),
                                                 1),
                                       np.array([[0, 0, 0, 1]]), 0))

    def multiply(self, b: 'Matrix') -> 'Matrix':
        return Matrix(matrix=np.matmul(self._matrix, b.to_array()))

    def negate(self) -> 'Matrix':
        return Matrix(matrix=np.negative(self._matrix))

    def rref(self) -> 'Matrix':
        rref = np.zeros((len(self._matrix), len(self._matrix[0])))
        for i in range(len(self._matrix)):
            rref[i] = self._matrix[i, :]

        r = 0
        for c in range(len(rref[0])):
            if r >= len(rref):
                break
            j = r
            for i in range(r + 1, len(rref)):
                if abs(rref[i, c]) > abs(rref[j, c]):
                    j = i
            if abs(rref[j, c]) < 0.00001:
                continue

            temp = np.copy(rref[j])
            rref[j] = np.copy(rref[r])
            rref[r] = np.copy(temp)

            s = 1.0 / rref[r, c]
            for j in range(len(rref[0])):
                rref[r, j] *= s
            for i in range(len(rref)):
                if i != r:
                    t = rref[i, c]
                    for j in range(len(rref[0])):
                        rref[i, j] -= t * rref[r, j]
            r += 1

        return Matrix(matrix=rref)

    def transpose(self) -> 'Matrix':
        return Matrix(matrix=self._matrix.transpose())

    def __eq__(self, other) -> bool:
        return isinstance(other, Matrix) and np.allclose(self._matrix, other._matrix, rtol=1.e-13, atol=1.e-13)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return self._matrix.__repr__()

    def __add__(self, other) -> npt.NDArray:
        return self._matrix + other.to_array()

    def __sub__(self, other) -> npt.NDArray:
        return self._matrix - other.to_array()
