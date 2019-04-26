import doctest
import unittest

import boolean


def load_tests(loader, tests, ignore):
    """Include doctests in our unit test suite."""
    tests.addTests(doctest.DocTestSuite(boolean))
    return tests


if __name__ == '__main__':
    unittest.main()
