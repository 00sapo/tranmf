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
        with open("W.pkl", "wb") as f:
            pickle.dump(W, f)
        with open("W.pkl", "rb") as f:
            pickle.load(f)

        # show the A, Z and g (hex codepoints 41, 5A and 67)
        Image.fromarray(W.get_glyph("0x41")).show()
        Image.fromarray(W.get_glyph("Z")).show()
        Image.fromarray(W.get_glyph("g")).show()
        # show the whole W after reshaping it to a 2D array (third dimension is the glyphs)
        Image.fromarray(W.glyphs.reshape(-1, W.glyphs.shape[2])).show()

        assert W.glyphs.shape[0] == 50
        maps = np.zeros(W.glyphs.shape[1], dtype=bool)
        for _, v in W.codemap.items():
            assert np.any(W.glyphs[:, v[0] : v[1]] != 0), "Some glyphs are empty."
            maps[v[0] : v[1]] = True
        assert all(maps), "Some glyphs are missing in the codemap."

    def test_run_single_nmf(self):
        image_file = THIS_DIR / "data/strip.png"
        image_strip = Image.open(image_file)

        W = pickle.load(open("W.pkl", "rb"))
        W = W.select_alphabet("abcdefghjklmnopqrstuvwxyzABCDEFCHIJKLMNOPQRSTUVWXYZ")
        W, H = run_single_nmf(image_strip, W)

        print(H.max(), H.min())

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
