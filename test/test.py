#!/usr/bin/env/ python3
import sys
import unittest

if __name__ == "__main__":
    sys.path.append('./src/')
    loader = unittest.TestLoader()
    suite = loader.discover('./test/')
    runner = unittest.TextTestRunner()
    runner.run(suite)
