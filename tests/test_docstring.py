from __future__ import print_function

import unittest
import sys
import os


sys.path.append(os.getcwd())
import numpyeigen_test as npe_test


class TestDocstring(unittest.TestCase):

    def test_docstring(self):
        docstr = "This is\n" + \
                 "a multi-line\n" + \
                 "documentation\n" + \
                 "string..."

        self.assertTrue(docstr in npe_test.docstring.__doc__)

        docstr = "this is the docstr"
        self.assertTrue(docstr in npe_test.string_expr_docstring.__doc__)


if __name__ == '__main__':
    unittest.main()
