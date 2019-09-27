from __future__ import print_function
import unittest
import sys
import os
import time

sys.path.append(os.getcwd())
import numpy as np
import numpyeigen_test as npe_test
import numpyeigen_helpers as npe_helpers
import platform
import scipy.sparse as sp


class TestDenseBindings(unittest.TestCase):

    def test_no_copy(self):
        a = np.eye(100)
        expected = np.array(a)
        expected[0, 0] = 2.0
        self.assertEqual(a[0, 0], 1.0)
        ret = npe_test.mutate_matrix(a)  # Always sets A[0, 0] to 2.0
        self.assertTrue(np.array_equal(ret, a))
        self.assertTrue(np.array_equal(a, expected))

    def test_pybind_does_a_copy_if_type_mismatch(self):
        a = np.eye(100)
        expected = np.array(a)
        expected[0, 0] = 2.0
        self.assertEqual(a[0, 0], 1.0)
        ret = npe_helpers.mutate_copy(a)  # Always sets A[0, 0] to 2.0
        self.assertFalse(np.array_equal(ret, a))
        self.assertFalse(np.array_equal(a, expected))

    def test_pybind_does_not_make_a_copy_if_type_mismatch(self):
        a = np.eye(100, dtype=np.float32)
        expected = np.array(a, dtype=np.float32)
        expected[0, 0] = 2.0
        self.assertEqual(a[0, 0], 1.0)
        ret = npe_helpers.mutate_copy(a)  # Always sets A[0, 0] to 2.0
        self.assertTrue(np.array_equal(a, expected))

        a = np.eye(100, dtype=np.float64)
        expected = np.array(a, dtype=np.float64)
        expected[0, 0] = 2.0
        self.assertEqual(a[0, 0], 1.0)
        ret = npe_helpers.mutate_copy(a)  # Always sets A[0, 0] to 2.0
        self.assertFalse(np.array_equal(a, expected))

    def test_timing_for_copy_vs_no_copy(self):
        mat_size = 10000
        num_iters = 10

        times_nocopy = []
        a = np.eye(mat_size)
        for i in range(num_iters):
            start_time = time.time()
            npe_test.mutate_matrix(a)
            end_time = time.time()
            times_nocopy.append(end_time-start_time)

        times_nocopy_pybind = []
        a = np.eye(mat_size, dtype=np.float32)
        for i in range(num_iters):
            start_time = time.time()
            npe_helpers.mutate_copy(a)
            end_time = time.time()
            times_nocopy_pybind.append(end_time-start_time)

        times_copy = []
        a = np.eye(mat_size)
        for i in range(num_iters):
            start_time = time.time()
            npe_helpers.mutate_copy(a)
            end_time = time.time()
            times_copy.append(end_time-start_time)

        median_nocopy = np.median(times_nocopy)
        median_copy = np.median(times_copy)
        self.assertLess(median_nocopy*1e3, median_copy)
        # print("COPY:")
        # print("  mean:", np.mean(times_copy))
        # print("  std:", np.std(times_copy))
        # print("  med:", np.median(times_copy))
        #
        # print("NOCOPY pybind22:")
        # print("  mean:", np.mean(times_nocopy))
        # print("  std:", np.std(times_nocopy))
        # print("  med:", np.median(times_nocopy))
        #
        # print("NOCOPY pybind11:")
        # print("  mean:", np.mean(times_nocopy_pybind))
        # print("  std:", np.std(times_nocopy_pybind))
        # print("  med:", np.median(times_nocopy_pybind))

    def test_return_does_not_copy(self):
        mat_size = 10000
        num_iters = 10

        times_nocopy = []
        a = np.eye(mat_size)
        for i in range(num_iters):
            start_time = time.time()
            npe_test.mutate_matrix(a)
            end_time = time.time()
            times_nocopy.append(end_time-start_time)

        times_copy = []
        a = np.eye(mat_size)
        for i in range(num_iters):
            start_time = time.time()
            npe_helpers.return_dense_copy(a)
            end_time = time.time()
            times_copy.append(end_time-start_time)

        median_nocopy = np.median(times_nocopy)
        median_copy = np.median(times_copy)

        print("COPY:")
        print("  mean:", np.mean(times_copy))
        print("  std:", np.std(times_copy))
        print("  med:", np.median(times_copy))

        print("NOCOPY pybind22:")
        print("  mean:", np.mean(times_nocopy))
        print("  std:", np.std(times_nocopy))
        print("  med:", np.median(times_nocopy))

        self.assertLess(median_nocopy*1e3, median_copy)

    def test_bool_array(self):
        a = np.zeros(10, dtype=np.bool)
        a[np.random.rand(10) > 0.5] = True
        b = np.zeros(10, dtype=np.bool)
        b[np.logical_not(a)] = True

        c = npe_test.bool_array(a, b)

        self.assertTrue(np.array_equal(c, np.ones(10, dtype=np.bool)))

        a = np.zeros((10, 10), dtype=np.bool)
        a[np.random.rand(10, 10) > 0.5] = True
        b = np.zeros((10, 10), dtype=np.bool)
        b[np.logical_not(a)] = True

        c = npe_test.bool_array(a, b)

        self.assertTrue(np.array_equal(c, np.ones((10, 10), dtype=np.bool)))

    def test_long_and_int(self):
        if sys.version_info[0] >= 3:
            along = np.ones((10, 10), dtype="long")
        aint = np.ones((10, 10), dtype="int32")
        alonglong = np.ones((10, 10), dtype="longlong")

        if sys.version_info[0] >= 3:
            blong = np.ones((10, 10), dtype="long")
        bint = np.ones((10, 10), dtype="int32")
        blonglong = np.ones((10, 10), dtype="longlong")

        is_64bits = sys.maxsize > 2 ** 32
        if not is_64bits:
            raise ValueError("Numpyeigen does not work on 32 bit systems yet!")

        if platform.system() != 'Windows':
            npe_test.intlonglong(alonglong, bint)
            if sys.version_info[0] >= 3:
                npe_test.intlonglong(along, bint)
                npe_test.intlonglong(alonglong, blong)
                npe_test.intlonglong(along, blong)
                npe_test.intlonglong(along, blonglong)
                npe_test.intlonglong(aint, blong)
            npe_test.intlonglong(aint, bint)
            npe_test.intlonglong(alonglong, blonglong)
            npe_test.intlonglong(aint, blonglong)

            if sys.version_info[0] >= 3:
                npe_test.intlong(along, bint)
                npe_test.intlong(alonglong, blong)
                npe_test.intlong(along, blong)
                npe_test.intlong(aint, blong)
                npe_test.intlong(along, blonglong)
            npe_test.intlong(alonglong, bint)
            npe_test.intlong(aint, bint)
            npe_test.intlong(alonglong, blonglong)
            npe_test.intlong(aint, blonglong)

            with self.assertRaises(ValueError):
                if sys.version_info[0] >= 3:
                    npe_test.longlonglong(along, bint)
                    npe_test.longlonglong(aint, blong)
                npe_test.longlonglong(alonglong, bint)
                npe_test.longlonglong(aint, bint)
                npe_test.longlonglong(aint, blonglong)

            if sys.version_info[0] >= 3:
                npe_test.longlonglong(alonglong, blong)
                npe_test.longlonglong(along, blong)
                npe_test.longlonglong(along, blonglong)
            npe_test.longlonglong(alonglong, blonglong)
        else:
            if sys.version_info[0] >= 3:
                npe_test.intlonglong(along, bint)
                npe_test.intlonglong(alonglong, blong)
                npe_test.intlonglong(along, blong)
                npe_test.intlonglong(along, blonglong)
                npe_test.intlonglong(aint, blong)
            npe_test.intlonglong(alonglong, bint)
            npe_test.intlonglong(aint, bint)
            npe_test.intlonglong(alonglong, blonglong)
            npe_test.intlonglong(aint, blonglong)

            with self.assertRaises(ValueError):
                if sys.version_info[0] >= 3:
                    npe_test.intlong(alonglong, blong)
                    npe_test.intlong(along, blonglong)
                npe_test.intlong(alonglong, bint)
                npe_test.intlong(alonglong, blonglong)
                npe_test.intlong(aint, blonglong)

            if sys.version_info[0] >= 3:
                npe_test.intlong(along, bint)
                npe_test.intlong(along, blong)
                npe_test.intlong(aint, blong)
            npe_test.intlong(aint, bint)

            if sys.version_info[0] >= 3:
                npe_test.longlonglong(alonglong, blong)
                npe_test.longlonglong(along, blonglong)
                npe_test.longlonglong(along, bint)
                npe_test.longlonglong(along, blong)
                npe_test.longlonglong(aint, blong)
            npe_test.longlonglong(alonglong, bint)
            npe_test.longlonglong(aint, blonglong)
            npe_test.longlonglong(aint, bint)
            npe_test.longlonglong(alonglong, blonglong)

    def test_dense_like(self):
        a = sp.diags([np.ones(100)], [0], format="csr")
        b1 = np.eye(100, dtype=np.float64)
        b2 = np.eye(100, dtype=np.float32)
        c1 = np.eye(100, dtype=np.float64)
        c2 = np.eye(100, dtype=np.float32)

        ret = npe_test.dense_like_1(a, b1)
        val = (ret - b1)
        self.assertEqual(np.linalg.norm(val), 0.0)

        ret = npe_test.dense_like_2(a, b1, c1)
        val = (ret - b1)
        self.assertEqual(np.linalg.norm(val), 0.0)
        val = (ret - c1)
        self.assertEqual(np.linalg.norm(val), 0.0)

        ret = npe_test.dense_like_3(a, b1, c1)
        val = (ret - b1)
        self.assertEqual(np.linalg.norm(val), 0.0)
        val = (ret - c1)
        self.assertEqual(np.linalg.norm(val), 0.0)

        ret = npe_test.dense_like_4(b1, b1, c1)
        val = (ret - b1)
        self.assertEqual(np.linalg.norm(val), 0.0)
        val = (ret - c1)
        self.assertEqual(np.linalg.norm(val), 0.0)

        with self.assertRaises(ValueError):
            npe_test.dense_like_1(a, b2)
        with self.assertRaises(ValueError):
            npe_test.dense_like_2(a, b1, c2)
        with self.assertRaises(ValueError):
            npe_test.dense_like_2(a, b2, c1)
        with self.assertRaises(ValueError):
            npe_test.dense_like_2(a, b2, c2)
        with self.assertRaises(ValueError):
            npe_test.dense_like_3(a, b1, c2)
        with self.assertRaises(ValueError):
            npe_test.dense_like_3(a, b2, c1)
        with self.assertRaises(ValueError):
            npe_test.dense_like_3(a, b2, c2)


if __name__ == '__main__':
    unittest.main()
