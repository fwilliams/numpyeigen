import os
import sys
import unittest

sys.path.append(os.getcwd())
import numpy as np
import numpyeigen_test as npe_test


class TestNpeCallInterface(unittest.TestCase):

    def test_default_arguments(self):
        a = np.eye(10)
        expected = np.array(a)
        expected[0, 0] = 2.0
        self.assertEqual(a[0, 0], 1.0)
        def_str, def_nparr, ret = npe_test.default_arg(a)
        self.assertEqual(def_str, "abcdef")
        self.assertEqual(def_nparr.shape, (0, 0))
        self.assertTrue(np.array_equal(ret, a))
        self.assertTrue(np.array_equal(a, expected))

        def_str, def_nparr, ret = npe_test.default_arg(a, doubleit=True)
        self.assertEqual(def_str, "abcabcdef")
        self.assertEqual(def_nparr.shape, (0, 0))
        self.assertTrue(np.array_equal(ret, a))
        self.assertTrue(np.array_equal(a, expected))

    def test_passing_subset_of_arguments(self):
        a = np.eye(100)
        expected = np.array(a)
        expected[0, 0] = 2.0
        self.assertEqual(a[0, 0], 1.0)

        def_str, def_nparr, ret = npe_test.default_arg(a, b="fff")
        self.assertEqual(def_str, "fffdef")
        self.assertEqual(def_nparr.shape, (0, 0))
        self.assertTrue(np.array_equal(ret, a))
        self.assertTrue(np.array_equal(a, expected))

        def_str, def_nparr, ret = npe_test.default_arg(a, c=np.eye(7))
        self.assertEqual(def_str, "abcdef")
        self.assertEqual(def_nparr.shape, (7, 7))
        self.assertTrue(np.array_equal(ret, a))
        self.assertTrue(np.array_equal(a, expected))

        def_str, def_nparr, ret = npe_test.default_arg(a, c=np.eye(7), b="fff")
        self.assertEqual(def_str, "fffdef")
        self.assertEqual(def_nparr.shape, (7, 7))
        self.assertTrue(np.array_equal(ret, a))
        self.assertTrue(np.array_equal(a, expected))

        def_str, def_nparr, ret = npe_test.default_arg(a, doubleit=True, c=np.eye(7), b="fff")
        self.assertEqual(def_str, "ffffffdef")
        self.assertEqual(def_nparr.shape, (7, 7))
        self.assertTrue(np.array_equal(ret, a))
        self.assertTrue(np.array_equal(a, expected))

    def test_passing_no_numpy_arguments(self):
        ret = npe_test.no_numpy("abc")
        self.assertEqual(ret, "abc")

    def test_dtype(self):
        a = np.zeros(10, dtype=np.float32)

        b = npe_test.test_dtype(a, dtype="float32")
        self.assertEqual(b.dtype, np.float32)

        b = npe_test.test_dtype(a, dtype="float64")
        self.assertEqual(b.dtype, np.float64)

        b = npe_test.test_dtype(a, dtype=np.float32)
        self.assertEqual(b.dtype, np.float32)

        b = npe_test.test_dtype(a, dtype=np.float64)
        self.assertEqual(b.dtype, np.float64)

        threw = False
        try:
            npe_test.test_dtype(a, dtype=np.int32)
        except TypeError:
            threw = True
        self.assertTrue(threw)

        threw = False
        try:
            npe_test.test_dtype(a, dtype="not_a_type")
        except TypeError:
            threw = True
        self.assertTrue(threw)

    def test_multiple_functions_per_file(self):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        ma1 = npe_test.matrix_add(a, b)
        ma2 = npe_test.matrix_add2(a, b)
        ma3 = npe_test.matrix_add3(a, b)
        print(ma1, ma2, ma3)
        self.assertTrue(np.array_equal(ma1, ma2))
        self.assertTrue(np.array_equal(ma2, ma3))
        self.assertTrue(np.array_equal(ma3, ma1))


if __name__ == '__main__':
    unittest.main()
