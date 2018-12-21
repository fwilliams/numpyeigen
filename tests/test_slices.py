import os
import sys
import unittest

sys.path.append(os.getcwd())
import numpy as np
import numpyeigen_test as npe_test


class TestSlices(unittest.TestCase):

    def test_passing_slices(self):
        def run_test(dtype1, dtype2, zero_thresh):
            a = np.random.rand(10, 11).astype(dtype1)
            b = np.random.rand(10, 11).astype(dtype2)

            res1 = npe_test.matrix_add4(a[0:10:2], b[0:10:2])
            res2 = a[0:10:2] + b[0:10:2]
            self.assertLessEqual(np.linalg.norm(res1-res2), zero_thresh)

            res1 = npe_test.matrix_add4(a[0:10:2], a[0:10:2])
            res2 = a[0:10:2] + a[0:10:2]
            self.assertLessEqual(np.linalg.norm(res1-res2), zero_thresh)

            res1 = npe_test.matrix_add4(a[0:10:2, 0:9:3], b[0:10:2, 0:9:3])
            res2 = a[0:10:2, 0:9:3] + b[0:10:2, 0:9:3]
            self.assertLessEqual(np.linalg.norm(res1-res2), zero_thresh)

            res1 = npe_test.matrix_add4(a[0:10:2, 0:9:3], a[0:10:2, 0:9:3])
            res2 = a[0:10:2, 0:9:3] + a[0:10:2, 0:9:3]
            self.assertLessEqual(np.linalg.norm(res1-res2), zero_thresh)

        run_test(np.float32, np.float32, 0.0)
        run_test(np.float64, np.float64, 0.0)
        run_test(np.float32, np.float64, 1e-6)
        run_test(np.float64, np.float32, 1e-6)

    def test_mixing_slices_and_non_slices(self):
        def run_test(dtype1, dtype2, zero_thresh):
            a = np.random.rand(10, 11).astype(dtype1)
            c = np.random.rand(*a[0:10:2].shape).astype(dtype2)
            d = np.random.rand(*a[0:10:2, 0:9:3].shape).astype(dtype2)

            res1 = npe_test.matrix_add4(a[0:10:2], c)
            res2 = a[0:10:2] + c
            self.assertLessEqual(np.linalg.norm(res1-res2), zero_thresh)

            res1 = npe_test.matrix_add4(c, a[0:10:2])
            res2 = a[0:10:2] + c
            self.assertLessEqual(np.linalg.norm(res1-res2), zero_thresh)

            res1 = npe_test.matrix_add4(a[0:10:2, 0:9:3], d)
            res2 = a[0:10:2, 0:9:3] + d
            self.assertLessEqual(np.linalg.norm(res1-res2), zero_thresh)

            res1 = npe_test.matrix_add4(d, a[0:10:2, 0:9:3])
            res2 = a[0:10:2, 0:9:3] + d
            self.assertLessEqual(np.linalg.norm(res1-res2), zero_thresh)

        run_test(np.float32, np.float32, 0.0)
        run_test(np.float64, np.float64, 0.0)
        run_test(np.float32, np.float64, 1e-6)
        run_test(np.float64, np.float32, 1e-6)


if __name__ == '__main__':
    unittest.main()
