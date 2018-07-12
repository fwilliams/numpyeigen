import unittest
import sys
import os
import time

sys.path.append(os.getcwd())
import numpy as np
import test_module as m


class TestSomething(unittest.TestCase):

    def test_no_copy(self):
        a = np.eye(100)
        expected = np.array(a)
        expected[0, 0] = 2.0
        self.assertEqual(a[0, 0], 1.0)
        ret = m.mutate_matrix(a)  # Always sets A[0, 0] to 2.0
        self.assertEqual(ret, 200)  # Always returns num_rows + num_cols
        self.assertTrue(np.array_equal(a, expected))

    def test_pybind_does_a_copy_if_type_mismatch(self):
        a = np.eye(100)
        expected = np.array(a)
        expected[0, 0] = 2.0
        self.assertEqual(a[0, 0], 1.0)
        ret = m.mutate_copy(a)  # Always sets A[0, 0] to 2.0
        self.assertEqual(ret, 200)  # Always returns num_rows + num_cols
        self.assertFalse(np.array_equal(a, expected))

    def test_pybind_does_not_make_a_copy_if_type_mismatch(self):
        a = np.eye(100, dtype=np.float32)
        expected = np.array(a, dtype=np.float32)
        expected[0, 0] = 2.0
        self.assertEqual(a[0, 0], 1.0)
        ret = m.mutate_copy(a)  # Always sets A[0, 0] to 2.0
        self.assertEqual(ret, 200)  # Always returns num_rows + num_cols
        self.assertTrue(np.array_equal(a, expected))

    def test_timing_for_copy_vs_no_copy(self):
        mat_size = 10000
        num_iters = 10

        times_nocopy = []
        a = np.eye(mat_size)
        for i in range(num_iters):
            start_time = time.time()
            m.mutate_matrix(a)
            end_time = time.time()
            times_nocopy.append(end_time-start_time)

        times_nocopy_pybind = []
        a = np.eye(mat_size, dtype=np.float32)
        for i in range(num_iters):
            start_time = time.time()
            m.mutate_copy(a)
            end_time = time.time()
            times_nocopy_pybind.append(end_time-start_time)

        times_copy = []
        a = np.eye(mat_size)
        for i in range(num_iters):
            start_time = time.time()
            m.mutate_copy(a)
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


if __name__ == '__main__':

    print("I AM A TESTTT")
    print(os.getcwd())
    unittest.main()
