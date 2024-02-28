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
        v = np.random.rand(50, 100)
        w = np.random.rand(50, 50)
        h = np.random.rand(50, 100)

        nmf = NMF(
            h,
            w,
            v,
            [Euclidean2D(1e-3, 1e-3)],
            "multiplicative",
            alternate=lambda x: True,
            verbose=True,
        )
        nmf.fit(100, 0.1, False, True)
        print(nmf._loss)


if __name__ == "__main__":
    unittest.main()
