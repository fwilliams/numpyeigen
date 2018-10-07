import unittest
import sys
import os

sys.path.append(os.getcwd())
import numpy as np
import numpyeigen_test as npe_test


class Test1dArays(unittest.TestCase):

    def test_passing_1d_arrays(self):
        a = np.ones(10)
        expected = np.array(a)
        expected[0] = 2.0
        self.assertEqual(a[0], 1.0)
        def_str, def_nparr, ret = npe_test.default_arg(a)
        self.assertEqual(def_str, "abcdef")
        self.assertEqual(def_nparr.shape, (0, 0))
        self.assertTrue(np.array_equal(ret, expected))
        self.assertTrue(np.array_equal(a, expected))


if __name__ == '__main__':
    unittest.main()
