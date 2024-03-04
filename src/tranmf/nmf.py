import shutil
import subprocess
import tempfile
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

THIS_DIR = Path(__file__).parent


@dataclass
class W:
    #: the W matrix (H x glyphs)
    array: np.ndarray

    #: the list of hex codes and (initial, end) positions
    codemap: dict[str, tuple[int, int]]

    def select_alphabet(self, alphabet: str) -> object:
        """Select a subset of the W matrix that is limited to the given alphabet."""
        new_array = []
        new_codemap = {}
        index = 0
        for k in alphabet:
            v = self.codemap[hex(ord(k))]
            new_array.append(self.array[:, v[0] : v[1]])
            new_codemap[k] = (index, index + v[1] - v[0])
            index += v[1] - v[0]

        return W(np.concatenate(new_array, axis=1), new_codemap)


def _force_flatpak_tmpdir_permissions(flatpak_name: str):
    tmpdir = tempfile.gettempdir()
    check_command = ["flatpak", "info", "--file-access=" + tmpdir, flatpak_name]
    # get output ithout printing it
    # subprocess.run(check_command, stdout=subprocess.PIPE)
    check = subprocess.run(check_command, capture_output=True, text=True)
    if check.returncode != 0 or "read" not in check.stdout.strip():
        subprocess.run(
            ["flatpak", "override", "--user", "--filesystem=" + tmpdir, flatpak_name]
        )


def _check_bin_or_flatpak(bin_name: str, flatpak_name: str):
    """
    Check if binary is available in PATH or as flatpak. Returns a list usable as
    subprocess.run() command.
    This function also forces flatpak apps to access the OS temporary directory.
    """
    if not shutil.which(bin_name):
        # Check if binary is available as flatpak
        bin_flatpaks = subprocess.run(
            ["flatpak", "list", "--app", "--columns=application"],
            capture_output=True,
            text=True,
        )
        if bin_flatpaks.returncode == 0 and flatpak_name in bin_flatpaks.stdout.split(
            "\n"
        ):
            _force_flatpak_tmpdir_permissions(flatpak_name)
            return ["flatpak", "run", flatpak_name]
        else:
            raise FileNotFoundError(f"{bin_name} not found in PATH.")
    else:
        return [bin_name]


def check_deps():
    """Check if inkscape and fontforge are available in PATH or as flatpak. Returns a
    NamedTuple with the paths to the binaries."""

    inkscape = _check_bin_or_flatpak("inkscape", "org.inkscape.Inkscape")
    fontforge = _check_bin_or_flatpak("fontforge", "org.fontforge.FontForge")
    Binaries = namedtuple("Binaries", "inkscape fontforge")
    return Binaries(inkscape=inkscape, fontforge=fontforge)


def build_initial_w(font_path: Union[Path, str], height=50):
    """Build initial W matrix from font file.
    It first runs `fontdump` bash script to extract the font file and then
    builds the matrix from the extracted font file.

    Args:
        font_path (Path or str): Path to the font file.
        height (int): Height of W.
    """
    W_dict = {}
    w = []
    cur_width = 0

    bins = check_deps()

    # Create temporary directory with a random name
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use fontforge to convert the font to SVG
        print("Converting font to SVG")
        subprocess.run(
            bins.fontforge
            + [
                "-lang=ff",
                "-c",
                f'Open("{font_path}"); SelectWorthOutputting(); foreach Export("{tmpdir}/%u-%e-%n.svg"); endloop;',  # %u is the hex Unicode point in hex, %e is the same in decimal
                str(font_path.resolve()),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Get the list of SVG files
        svg_files = list(Path(tmpdir).glob("*.svg"))
        for file in tqdm(svg_files, "Converting SVG to PNG"):
            try:
                outfile = file.with_suffix(".png")
                # Use inkscape to convert SVG to PNG
                subprocess.run(
                    bins.inkscape
                    + [
                        str(file.resolve()),
                        "--export-type=png",
                        f"--export-filename={outfile}",
                        "--export-area-drawing",
                        f"--export-height={height}",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # open the png file and append it to W
                w.append(np.asarray(Image.open(outfile)))
                # add the hex code to the dictionary
                hex_ = hex(int(file.stem.split("-")[1]))
                W_dict[hex_] = cur_width, cur_width + w[-1].shape[1]
                cur_width += w[-1].shape[1]
            except Exception as e:
                print(f"Error converting {file}: {e}")
                continue

        # create a numpy array from the images
        w = np.concatenate(w, axis=1)
        return W(w, W_dict)


def _setup_array(arr: np.ndarray, height: int) -> np.ndarray:
    """Resize the array to match the given height, converts to grayscale, and put the array in 0-1."""
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    ratio = height / arr.shape[0]
    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    arr = cv2.resize(arr, (round(arr.shape[1] * ratio), height), interpolation=interp)
    if arr.dtype in [np.uint8, np.uint16]:
        arr = arr.astype(np.float64) / 255
    return arr


def run_single_nmf(
    image_strip: Image.Image, W: W, n_iter=50, height=10
) -> tuple[W, np.ndarray]:
    """Run NMF on a single image strip.
    Args:
        image_strip (np.ndarray): The image strip to run NMF on.
        W (W): The W matrix.
        n_iter (int): Number of iterations.
    Returns:
        np.ndarray: The H matrix.
    """
    from tranmf._nmf import NMF, Diagonalize2D, Euclidean2D

    # put image in grayscale mode
    # resize image strip to match W's height
    # convert to float64 array in 0-1
    image_strip = _setup_array(np.array(image_strip), height)
    w = _setup_array(W.array, height)
    h = np.random.rand(w.shape[1], image_strip.shape[1])  # random H

    print("w shape", w.shape)
    print("h shape", h.shape)
    print("image_strip shape", image_strip.shape)

    nmf = NMF(
        [(0.5, Euclidean2D(1, 1)), (0.5, Diagonalize2D(1, 1))],
        "multiplicative",
        alternate=lambda x: False,
        verbose=True,
    )
    nmf.fit(
        w,
        h,
        image_strip,
        n_iter,
        0.1,  # loss tolerance
        fix_h=False,
        fix_w=True,
    )
    if nmf.get_loss() < 0.1:
        return W, h
    #
    # nmf.set_alternate(lambda x: True)
    # nmf.fit(
    #     w,
    #     h,
    #     image_strip,
    #     n_iter,
    #     0.1,  # loss tolerance
    #     fix_h=False,
    #     fix_w=True,
    # )
    W.array = w
    return W, h
