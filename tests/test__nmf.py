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
        v = np.random.rand(50, 512)
        w = np.random.rand(50, 1000)
        h = np.random.rand(1000, 512)

        nmf = NMF(
            [Euclidean2D(1e-3, 1e-3)],
            "multiplicative",
            alternate=lambda x: False,
            verbose=True,
        )
        nmf.fit(w, h, v, 5, 0.1, False, True)
        nmf.fit(w, h, v, 3, 0.1, True, False)
        nmf.set_alternate(lambda x: True)
        nmf.fit(w, h, v, 3, 0.1, False, True)


if __name__ == "__main__":
    unittest.main()
