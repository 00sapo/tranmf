import pickle
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).parent
sys.path.append(str(THIS_DIR.parent / "src/"))
from tranmf.nmf import build_initial_w, run_single_nmf


# 1. build_initial_w
# Use the test font_file in `./tests/data/BalladeHf.ttf`
# it then checks that the W matrix sub-matrices correspond to the glyphs in the
# font file.
class TestNMF(unittest.TestCase):
    def test_build_initial_w(self):
        font_file = THIS_DIR / "data/BalladeHf.ttf"
        W = build_initial_w(font_file, height=50)
        # pickle.dump(W, open("W.pkl", "wb"))
        # W = pickle.load(open("W.pkl", "rb"))

        # show the A, Z and g (hex codepoints 41, 5A and 67)
        Image.fromarray(W.array[:, W.codemap["0x41"][0] : W.codemap["0x41"][1]]).show()
        Image.fromarray(W.array[:, W.codemap["0x5a"][0] : W.codemap["0x5a"][1]]).show()
        Image.fromarray(W.array[:, W.codemap["0x67"][0] : W.codemap["0x67"][1]]).show()
        # show the whole W
        Image.fromarray(W.array).show()

        assert W.array.shape[0] == 50
        maps = np.zeros(W.array.shape[1], dtype=bool)
        for k, v in W.codemap.items():
            assert np.any(W.array[:, v[0] : v[1]] != 0), "Some glyphs are empty."
            maps[v[0] : v[1]] = True
        assert all(maps), "Some glyphs are missing in the codemap."

    def test_run_single_nmf(self):
        image_file = THIS_DIR / "data/strip.png"
        image_strip = Image.open(image_file)

        W = pickle.load(open("W.pkl", "rb"))
        W = W.select_alphabet("abcdefghjklmnopqrstuvwxyzABCDEFCHIJKLMNOPQRSTUVWXYZ")
        W, H = run_single_nmf(image_strip, W)

        # show the H matrix
        cv2.imshow("H", H)
        # show the W matrix
        cv2.imshow("W", W.array)
        # show the reconstructed image
        cv2.imshow("reconstructed", W.array @ H)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    unittest.main()
