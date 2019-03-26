from __future__ import print_function
import unittest
import sys
import os
import scipy as sp
import scipy.sparse

sys.path.append(os.getcwd())
import numpy as np
import numpyeigen_test as npe_test
import numpyeigen_helpers as npe_helpers

import platform


class TestDefaultMatches(unittest.TestCase):

    def test_not_none_1(self):
        a = np.eye(100)
        b = np.zeros([100, 100])
        c = np.zeros([100, 100])

        d, e, f = npe_test.default_matches_1(a, b, c)

        self.assertEqual(a[0, 0], 2.0)
        self.assertEqual(c[0, 0], 2.0)

        self.assertNotEqual(f[0, 0], 3.0)
        self.assertNotEqual(c[0, 0], 3.0)
        f[0, 0] = 3.0
        self.assertEqual(f[0, 0], 3.0)
        self.assertEqual(f[0, 0], c[0, 0])

    def test_none_1(self):
        a = np.eye(100)
        b = np.zeros([100, 100])

        d, e, f = npe_test.default_matches_1(a, b)

        self.assertEqual(a[0, 0], 2.0)
        self.assertEqual(f.shape, (0, 0))
        self.assertEqual(f.dtype, a.dtype)

    def test_not_none_2(self):
        a = np.random.rand(25, 25).astype(np.float32)
        b = np.random.rand(22, 21).astype(np.float32)
        c = np.random.rand(32, 33).astype(np.float32)

        d = np.eye(100, dtype=np.int32)
        e = np.zeros([100, 100], dtype=np.int32)
        f = np.zeros([100, 100], dtype=np.int32)

        g, h, i = npe_test.default_matches_2(a, b, c, d, e, f)

        self.assertEqual(d[0, 0], 2.0)
        self.assertEqual(e[0, 0], 2.0)
        self.assertEqual(f[0, 0], 2.0)

        self.assertNotEqual(f[0, 0], 3.0)
        self.assertNotEqual(i[0, 0], 3.0)
        f[0, 0] = 3.0
        self.assertEqual(i[0, 0], 3.0)
        self.assertEqual(f[0, 0], i[0, 0])

        self.assertNotEqual(e[0, 0], 3.0)
        self.assertNotEqual(h[0, 0], 3.0)
        e[0, 0] = 3.0
        self.assertEqual(e[0, 0], 3.0)
        self.assertEqual(e[0, 0], h[0, 0])

        self.assertNotEqual(d[0, 0], 3.0)
        self.assertNotEqual(g[0, 0], 3.0)
        g[0, 0] = 3.0
        self.assertEqual(d[0, 0], 3.0)
        self.assertEqual(d[0, 0], g[0, 0])

    def test_none_2(self):
        if platform.system() == 'Windows':
            print("Warning skipping test on windows")
            return

        a = np.random.rand(25, 25).astype(np.float32)
        b = np.random.rand(22, 21).astype(np.float32)

        d = np.eye(100, dtype=np.int32)
        e = np.zeros([100, 100], dtype=np.int32)

        g, h, i = npe_test.default_matches_2(a, b, None, d, e, None)

        self.assertEqual(d[0, 0], 2.0)
        self.assertEqual(e[0, 0], 2.0)
        self.assertEqual(i.shape, (0, 0))
        self.assertEqual(i.dtype, d.dtype)

        self.assertNotEqual(e[0, 0], 3.0)
        self.assertNotEqual(h[0, 0], 3.0)
        e[0, 0] = 3.0
        self.assertEqual(e[0, 0], 3.0)
        self.assertEqual(e[0, 0], h[0, 0])

        self.assertNotEqual(d[0, 0], 3.0)
        self.assertNotEqual(g[0, 0], 3.0)
        g[0, 0] = 3.0
        self.assertEqual(d[0, 0], 3.0)
        self.assertEqual(d[0, 0], g[0, 0])

    def test_not_none_3(self):
        print("NOT NONE 3")
        a = np.random.rand(25, 25).astype(np.float32)
        a[a < 0.5] = 0.0
        b = np.random.rand(22, 21).astype(np.float32)
        b[b < 0.5] = 0.0
        c = np.random.rand(32, 33).astype(np.float32)
        c[c < 0.5] = 0.0

        a = sp.sparse.csr_matrix(a)
        b = sp.sparse.csr_matrix(b)
        c = sp.sparse.csr_matrix(c)
        d = np.eye(100, dtype=np.int32)
        e = np.zeros([100, 100], dtype=np.int32)
        f = np.zeros([100, 100], dtype=np.int32)

        print("ABOUT TO CALL")
        g, h, i = npe_test.default_matches_3(a, b, c, d, e, f)

        self.assertEqual(type(g), sp.sparse.csr_matrix)
        self.assertEqual(type(h), sp.sparse.csr_matrix)
        self.assertEqual(type(i), sp.sparse.csr_matrix)

    def test_none_3(self):
        a = np.random.rand(25, 25).astype(np.float32)
        a[a < 0.5] = 0.0
        b = np.random.rand(22, 21).astype(np.float32)
        b[b < 0.5] = 0.0
        c = np.random.rand(32, 33).astype(np.float32)
        c[c < 0.5] = 0.0

        a = sp.sparse.csr_matrix(a)
        b = sp.sparse.csr_matrix(b)
        d = np.eye(100, dtype=np.int32)

        sys.stdout.flush()
        g, h, i = npe_test.default_matches_3(a, b, None, d)

        self.assertEqual(type(g), np.ndarray)
        self.assertEqual(type(h), np.ndarray)
        self.assertEqual(type(i), np.ndarray)


if __name__ == '__main__':
    unittest.main()
