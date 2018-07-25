import unittest
import sys
import os
import time

sys.path.append(os.getcwd())
import numpy as np
import scipy.sparse as sp
import sparse_test as st
import numpyeigen_test as npe_test


class TestSparseMatrixWrapper(unittest.TestCase):

    def test_sparse_matrix_wrapper(self):
        a = sp.csr_matrix(np.eye(100))
        b = st.test(a)
        self.assertTrue(a is b)

    def test_sparse_matrix_binding(self):
        a = sp.csr_matrix(np.eye(100))
        b = sp.csr_matrix(np.eye(100))
        ret = npe_test.sparse_matrix_add(a, b)
        self.assertEqual(ret, (100, 100))


if __name__ == '__main__':
    unittest.main()
