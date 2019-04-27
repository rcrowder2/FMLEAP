import doctest
import unittest

import evolve


def load_tests(loader, tests, ignore):
    """Include doctests in our unit test suite."""
    tests.addTests(doctest.DocTestSuite(evolve))
    return tests


if __name__ == '__main__':
    unittest.main()
