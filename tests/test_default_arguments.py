import unittest
import sys
import os
import time

sys.path.append(os.getcwd())
import numpy as np
import numpyeigen_test as npe_test
import numpyeigen_helpers as npe_helpers


class TestDefaultArguments(unittest.TestCase):

    def test_default_arguments(self):
        a = np.eye(100)
        expected = np.array(a)
        expected[0, 0] = 2.0
        self.assertEqual(a[0, 0], 1.0)
        def_str, def_nparr, ret = npe_test.default_arg(a)
        self.assertEqual(def_str, "abcdef")
        self.assertEqual(def_nparr.shape, (0,))
        self.assertTrue(np.array_equal(ret, a))
        self.assertTrue(np.array_equal(a, expected))

    def test_passing_subset_of_arguments(self):
        a = np.eye(100)
        expected = np.array(a)
        expected[0, 0] = 2.0
        self.assertEqual(a[0, 0], 1.0)

        def_str, def_nparr, ret = npe_test.default_arg(a, b="fff")
        self.assertEqual(def_str, "fffdef")
        self.assertEqual(def_nparr.shape, (0,))
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

if __name__ == '__main__':
    unittest.main()
