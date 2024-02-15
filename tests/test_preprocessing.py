import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).parent
sys.path.append(str(THIS_DIR.parent / "src/"))
from tranmf.preprocessing import preprocess_image


# 1. preprocess_image
# Use the test image in `./tests/data/lena.png` and show the result using opencv
class TestPreprocessing(unittest.TestCase):
    def test_preprocess_image(self):
        image = Image.open(THIS_DIR / "data/Parigi4.png")
        image, ink_mask = preprocess_image(
            np.asarray(image)[..., ::-1], debug_dir="tests/output"
        )
        cv2.imshow("Preprocessed image", image)
        cv2.waitKey(0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray image", image)
        cv2.waitKey(0)
        image[ink_mask == 0] = 255
        cv2.imshow("Gray ink", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    unittest.main()
