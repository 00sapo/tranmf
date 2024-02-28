import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).parent
sys.path.append(str(THIS_DIR.parent / "src/"))


class TestNMF(unittest.TestCase):
    def test_main(self):
        import numpy as np

        from tranmf._nmf import NMF, Euclidean2D

        # a little test with random data
        v = np.random.rand(10, 12)
        w = np.random.rand(10, 7)
        h = np.random.rand(7, 12)

        nmf = NMF(h, w, v, [Euclidean2D()], "multiplicative")
        nmf.fit(100, 1e-5, False, False)


if __name__ == "__main__":
    unittest.main()
