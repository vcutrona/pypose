import unittest

from pypose.matrix import Matrix
import numpy as np


class MatrixTestCase(unittest.TestCase):

    def setUp(self):
        print("init matrix")
        self._m = Matrix(matrix=np.array([[-0.48624914, 0.12857975, -1.15674627, 0.2721935],
                                          [1.40340288, -0.75898342, -0.94040306, -1.05091621],
                                          [0.22158037, -0.51416981, 0.57565827, 3.22888385],
                                          [0.00832453, -1.33374083, 1.39952405, 0.05180333]]))
        print(self._m)
        print()

    def test_negate(self):
        print("negate test")
        negate = self._m.negate()
        print(negate)
        print()
        self.assertEqual(negate, Matrix(matrix=[
            [0.48624914, -0.12857975, 1.15674627, -0.2721935],
            [-1.40340288, 0.75898342, 0.94040306, 1.05091621],
            [-0.22158037, 0.51416981, -0.57565827, -3.22888385],
            [-0.00832453, 1.33374083, -1.39952405, -0.05180333]]))

    def test_sub_matrix(self):
        print("sub matrix test")
        sub_matrix = self._m.sub_matrix(0, 2, 1, 3)
        print(sub_matrix)
        print()
        self.assertEqual(sub_matrix, Matrix(matrix=[[0.12857975, -1.15674627], [-0.75898342, -0.94040306]]))

    def test_determinant(self):
        print("determinant test")
        det = self._m.determinant()
        print(det)
        print()
        self.assertAlmostEqual(det, -10.180387816803243, delta=1e-10)

    def test_inverse(self):
        print("inverse test")
        inv = self._m.inverse()
        print(inv)
        print()
        self.assertEqual(inv, Matrix(matrix=[
            [-0.7250499478601501, 0.43088943365026194, 0.207704565777567, -0.3951742964554022],
            [-0.6517757791535331, -0.220826710624144, -0.005944506198388586, -0.6846494075315849],
            [-0.6189113047901054, -0.21201293125207385, -0.017919864136582277, 0.06789077000977069],
            [0.05630887533606187, -0.026935645504764388, 0.29769915007294184, -0.09400928812501025]]))

    def test_invert_trans(self):
        print("invert trans test")
        inv_t = self._m.invert_transformation()
        print(inv_t)
        print()
        self.assertEqual(inv_t, Matrix(matrix=[
            [-0.9617480511957851, 0.5441152113815323, -1.0436936155277567, 4.203566523447272],
            [-1.0618611963129496, -0.024660202993590376, -2.174023328889439, 7.282784724645101],
            [-0.5782466749307031, -0.23146506997688737, 0.1970697920219708, -0.7221708765573454],
            [0.0, 0.0, 0.0, 1.0]]))

    def test_multiply(self):
        print("multiply test")
        mul = self._m.multiply(self._m)
        print(mul)
        print()
        self.assertEqual(mul, Matrix(matrix=[
            [0.1628410340642547, 0.07161673399684365, 0.1566008815181948, -3.9883792188877214],
            [-1.9646862024753364, 2.641681744164049, -2.9217640251421493, -1.9112680911370417],
            [-0.6748971435804342, -4.183743219949366, 5.077497642731884, 2.6266627511983747],
            [-1.565285222351411, 0.22467231130341164, 2.1227721884034167, 5.92549992887309]]))

    def test_rref(self):
        print("rref test")
        rref = self._m.rref()
        print(rref)
        print()
        self.assertEqual(rref, Matrix(matrix=[
            [1.0, -1.0242804337187671E-16, 0.0, -1.1102230246251565E-16],
            [-0.0, 0.9999999999999999, 0.0, 3.469446951953614E-18],
            [-0.0, 1.8172991807598734E-17, 1.0, 6.938893903907228E-18],
            [0.0, -1.8088507788337076E-17, 0.0, 0.9999999999999999]]))

    def test_transpose(self):
        print("transpose test")
        t = self._m.transpose()
        print(t)
        print()
        self.assertEqual(t, Matrix(matrix=[
            [-0.48624914, 1.40340288, 0.22158037, 0.00832453],
            [0.12857975, -0.75898342, -0.51416981, -1.33374083],
            [-1.15674627, -0.94040306, 0.57565827, 1.39952405],
            [0.2721935, -1.05091621, 3.22888385, 0.05180333]]))


if __name__ == '__main__':
    unittest.main()
