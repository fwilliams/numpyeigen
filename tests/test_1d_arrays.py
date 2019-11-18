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

    def test_passing_1d_arrays_1(self):
        a = np.ones(10)
        v = np.ones([10, 10])
        f = np.ones([10, 10], dtype=np.int)

        retv, retp = npe_test.one_d_arg(v, f, a)
        self.assertTrue(np.array_equal(retp, a))
        self.assertTrue(np.array_equal(retv, v))

        a = np.ones([10, 10])
        v = np.ones([10, 10])
        f = np.ones([10, 10], dtype=np.int)

        retv, retp = npe_test.one_d_arg(v, f, a)
        self.assertTrue(np.array_equal(retp, a))
        self.assertTrue(np.array_equal(retv, v))

    def test_passing_1d_arrays_2(self):
        v = np.ones(10)
        f = np.ones([10, 10], dtype=np.int)
        p = np.ones(10)
        q = np.ones(10)
        r = np.ones(10)
        s = np.ones(10)

        retv, retp = npe_test.one_d_arg_big(v, f, p, q, r, s)
        self.assertTrue(np.array_equal(retp, p))
        self.assertTrue(np.array_equal(retv, v))

        with self.assertRaises(ValueError):
            v = np.ones(10, dtype=np.float32)
            f = np.ones([10, 10], dtype=np.int)
            p = np.ones(10)
            q = np.ones(10)
            r = np.ones(10)
            s = np.ones(10)
            npe_test.one_d_arg_big(v, f, p, q, r, s)

        with self.assertRaises(ValueError):
            v = np.ones(10)
            f = np.ones([10, 10], dtype=np.int)
            p = np.ones(10, dtype=np.float32)
            q = np.ones(10)
            r = np.ones(10)
            s = np.ones(10)
            npe_test.one_d_arg_big(v, f, p, q, r, s)

        with self.assertRaises(ValueError):
            v = np.ones(10)
            f = np.ones([10, 10], dtype=np.int)
            p = np.ones(10)
            q = np.ones(10, dtype=np.float32)
            r = np.ones(10)
            s = np.ones(10)
            npe_test.one_d_arg_big(v, f, p, q, r, s)

        with self.assertRaises(ValueError):
            v = np.ones(10)
            f = np.ones([10, 10], dtype=np.int)
            p = np.ones(10)
            q = np.ones(10)
            r = np.ones(10, dtype=np.float32)
            s = np.ones(10)
            npe_test.one_d_arg_big(v, f, p, q, r, s)

        with self.assertRaises(ValueError):
            v = np.ones(10)
            f = np.ones([10, 10], dtype=np.int)
            p = np.ones(10)
            q = np.ones(10)
            r = np.ones(10)
            s = np.ones(10, dtype=np.float32)
            npe_test.one_d_arg_big(v, f, p, q, r, s)

    def test_passing_0d_arrays(self):
        dim = np.random.randint(5)
        a = np.zeros([0, dim])
        expected = np.array([0, dim])
        ret = npe_test.default_arg2(a)
        self.assertTrue(np.array_equal(ret, expected))

    def test_passing_0d_arrays_1(self):
        dim = np.random.randint(5)
        # (np.zeros([0, dim]), np.zeros([dim, 0]), np.zeros([0]), np.zeros([0, 0])):

        a = np.zeros([0, dim])
        v = np.ones([10, 10])
        f = np.ones([10, 10], dtype=np.int)

        retv, retp = npe_test.one_d_arg(v, f, a)
        self.assertEqual(retp.shape, a.shape)
        self.assertEqual(len(a), 0)
        self.assertTrue(np.array_equal(retv, v))

        #
        np.zeros([dim, 0])
        v = np.ones([10, 10])
        f = np.ones([10, 10], dtype=np.int)

        retv, retp = npe_test.one_d_arg(v, f, a)
        self.assertEqual(retp.shape, a.shape)
        self.assertEqual(len(a), 0)
        self.assertTrue(np.array_equal(retv, v))

        #
        a = np.zeros([0])
        v = np.ones([10, 10])
        f = np.ones([10, 10], dtype=np.int)

        retv, retp = npe_test.one_d_arg(v, f, a)
        self.assertEqual(retp.shape, a.reshape([0, 0]).shape)
        self.assertEqual(len(a), 0)
        self.assertTrue(np.array_equal(retv, v))

        #
        a = np.zeros([0, 0])
        v = np.ones([10, 10])
        f = np.ones([10, 10], dtype=np.int)

        retv, retp = npe_test.one_d_arg(v, f, a)
        self.assertEqual(retp.shape, a.shape)
        self.assertEqual(len(a), 0)
        self.assertTrue(np.array_equal(retv, v))

        #
        a = np.zeros([])
        v = np.ones([10, 10])
        f = np.ones([10, 10], dtype=np.int)

        retv, retp = npe_test.one_d_arg(v, f, a)
        print(a.shape, retp.shape)
        self.assertEqual(tuple(retp.shape), (0, 0))
        self.assertTrue(np.array_equal(retv, v))

    def test_passing_0d_arrays_2(self):
        dim = np.random.randint(5)
        a = np.zeros([0, dim])

        for arr_test in (np.zeros([0, dim])):
            v = arr_test.copy()
            f = np.ones([10, 10], dtype=np.int)
            p = arr_test.copy()
            q = arr_test.copy()
            r = arr_test.copy()
            s = arr_test.copy()

            retv, retp = npe_test.one_d_arg_big(v, f, p, q, r, s)
            self.assertEqual(retp.shape, a.shape)
            self.assertEqual(len(a), 0)
            self.assertTrue(np.array_equal(retv, v))

            with self.assertRaises(ValueError):
                v = arr_test.astype(np.float32)  # np.ones(10, dtype=np.float32)
                f = np.ones([10, 10], dtype=np.int)
                p = arr_test.copy()
                q = arr_test.copy()
                r = arr_test.copy()
                s = arr_test.copy()
                npe_test.one_d_arg_big(v, f, p, q, r, s)

            with self.assertRaises(ValueError):
                v = arr_test.copy()
                f = np.ones([10, 10], dtype=np.int)
                p = arr_test.astype(np.float32)
                q = arr_test.copy()
                r = arr_test.copy()
                s = arr_test.copy()
                npe_test.one_d_arg_big(v, f, p, q, r, s)

            with self.assertRaises(ValueError):
                v = arr_test.copy()
                f = np.ones([10, 10], dtype=np.int)
                p = arr_test.copy()
                q = arr_test.astype(np.float32)
                r = arr_test.copy()
                s = arr_test.copy()
                npe_test.one_d_arg_big(v, f, p, q, r, s)

            with self.assertRaises(ValueError):
                v = arr_test.copy()
                f = np.ones([10, 10], dtype=np.int)
                p = arr_test.copy()
                q = arr_test.copy()
                r = arr_test.astype(np.float32)
                s = arr_test.copy()
                npe_test.one_d_arg_big(v, f, p, q, r, s)

            with self.assertRaises(ValueError):
                v = arr_test.copy()
                f = np.ones([10, 10], dtype=np.int)
                p = arr_test.copy()
                q = arr_test.copy()
                r = arr_test.copy()
                s = arr_test.astype(np.float32)
                npe_test.one_d_arg_big(v, f, p, q, r, s)


if __name__ == '__main__':
    unittest.main()
