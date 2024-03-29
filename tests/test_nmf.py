import pickle
import sys
import unittest
from pathlib import Path

import cv2
import jax
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

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
        cv2.imshow("A", W.get_glyph("0x41"))
        cv2.imshow("Z", W.get_glyph("Z"))
        cv2.imshow("g", W.get_glyph("g"))
        cv2.imshow("W", W.get_concatenated_w())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        assert len(W.glyphs) == len(
            W.codemap
        ), "The number of glyphs and the codemap do not match."
        maps = np.zeros(len(W.glyphs), dtype=bool)
        for _, v in W.codemap.items():
            assert np.any(W.glyphs[v] != 0), "Some glyphs are empty."
            maps[v] = True
        assert all(maps), "Some glyphs are missing in the codemap."

    def test_run_single_nmf(self):
        image_file = THIS_DIR / "data/strip.png"
        image_strip = Image.open(image_file).convert("L")
        th = threshold_otsu(np.array(image_strip))
        image_strip = np.array(image_strip) > th
        image_strip = image_strip.astype(np.float32)

        w = pickle.load(open("W.pkl", "rb"))
        w = w.select_alphabet("abcdefghjklmnopqrstuvwxyzABCDEFCHIJKLMNOPQRSTUVWXYZ")
        w_, h, codemap, wh = run_single_nmf(
            image_strip,
            w,
            max_iter=20000,
            tol=0.0001,
            # alternate=lambda x: True,
            patience=100,
            learning_rate=1e-1,
            freeze_w=True,
            freeze_h=False,
            verbose=True,
            network_params={
                "n_conv_layers": 3,
                "h_channel_size": 20,
                "kernel_size": (1, 9, 9),
                "activations": [jax.nn.relu, jax.nn.relu, jax.nn.relu],
            },
        )

        print(w_.min(), w_.max())
        print(h.min(), h.max())
        print(wh.min(), wh.max())

        # normalize the h and w matrices
        h = (h - h.min()) / (h.max() - h.min())
        w_ = (w_ - w_.min()) / (w_.max() - w_.min())

        # show the H matrix
        cv2.imshow("H", h)
        # cv2.imshow("H_2", (h > 0.5).astype(np.float32))
        # show the W matrix
        cv2.imshow("W", w_)
        # show the reconstructed image
        cv2.imshow("reconstructed", wh)
        cv2.imshow("original", np.asarray(image_strip))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    unittest.main()
