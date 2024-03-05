import shutil
import subprocess
import tempfile
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from PIL import Image
from torchnmf import nmf
from tqdm import tqdm

THIS_DIR = Path(__file__).parent


@dataclass
class W:
    #: the W matrix (H x glyphs)
    glyphs: list[np.ndarray]

    #: the list of hex codes and its index position
    codemap: dict[str, int]

    def get_stacked_w(self):
        """Create a new W matrix from a list of arrays and a codemap."""
        assert all(
            arr.ndim == 2 for arr in self.glyphs
        ), "Error, glyphs should have only 2 dimensions"
        max_width = max(arr.shape[1] for arr in self.glyphs)
        padded_arrays = [
            np.pad(
                arr,
                ((0, 0), (0, max_width - arr.shape[1])),
                mode="constant",
                constant_values=(255,),
            )
            for arr in self.glyphs
        ]
        return np.stack(padded_arrays, axis=1)

    def get_glyph(self, glyph: str) -> np.ndarray:
        """Get the glyph from the W matrix."""
        if len(glyph) == 1:
            try:
                glyph = hex(ord(glyph))
            except TypeError:
                raise ValueError("glyph must be a single character or a hex codepoint.")
        index = self.codemap[glyph]
        if index is None:
            raise ValueError(f"Glyph {glyph} not found in the W matrix.")
        elif index > len(self.glyphs):
            raise ValueError(f"Index {index} out of bounds.")
        return self.glyphs[index]

    def select_alphabet(self, alphabet: str) -> object:
        """Select a subset of the W matrix that is limited to the given alphabet."""
        new_arrays = []
        new_codemap = {}
        index = 0
        for k in alphabet:
            idx = self.codemap[hex(ord(k))]
            new_arrays.append(self.glyphs[idx])
            new_codemap[k] = index
            index += 1

        return W(new_arrays, new_codemap)


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


def process_svg_file(file_path, bins, height):
    try:
        outfile = file_path.with_suffix(".png")
        # Use inkscape to convert SVG to PNG
        subprocess.run(
            bins.inkscape
            + [
                str(file_path.resolve()),
                "--export-type=png",
                f"--export-filename={outfile}",
                "--export-area-drawing",
                f"--export-height={height}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Open the png file and return the image array and hex code
        image_array = np.asarray(Image.open(outfile))
        hex_code = hex(int(file_path.stem.split("-")[1]))
        return (hex_code, image_array)
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None


def build_initial_w(font_path: Union[Path, str], height=50) -> W:
    """Build initial W matrix from font file."""

    w_dict: dict[str, int] = {}
    w_arrays = []
    bins = check_deps()

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Converting font to SVG")
        subprocess.run(
            bins.fontforge
            + [
                "-lang=ff",
                "-c",
                f'Open("{font_path}"); SelectWorthOutputting(); foreach Export("{tmpdir}/%u-%e-%n.svg"); endloop;',
                str(font_path.resolve()),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Get the list of SVG files
        svg_files = list(Path(tmpdir).glob("*.svg"))

        results = Parallel(n_jobs=1)(
            delayed(process_svg_file)(file, bins, height) for file in tqdm(svg_files)
        )

        idx = 0
        for result in results:
            if result is not None:
                hex_code, image_array = result
                w_arrays.append(image_array)
                w_dict[hex_code] = idx
                idx += 1

    return W(w_arrays, w_dict)


def _setup_array(arr: np.ndarray, height: int) -> np.ndarray:
    """Resize the array to match the given height, converts to grayscale, and put the array in 0-1."""
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    ratio = height / arr.shape[0]
    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    arr = cv2.resize(arr, (round(arr.shape[1] * ratio), height), interpolation=interp)
    if arr.dtype in [np.uint8, np.uint16]:
        arr = arr.astype(np.float32) / 255
    return torch.tensor(arr)


def run_single_nmf(
    image_strip: np.ndarray, w: W, max_iter=50, height=50
) -> tuple[nmf.NMF, dict[str, int]]:
    """Run NMF on a single image strip.
    Args:
        image_strip (np.ndarray): The image strip to run NMF on.
        W (W): The W matrix.
        n_iter (int): Number of iterations.
    Returns:
        np.ndarray: The H matrix.
    """
    # from tranmf._nmf import NMF, Diagonalize2D, Euclidean2D

    # put image in grayscale mode
    # resize image strip to match W's height
    # convert to float64 array in 0-1
    image_strip = _setup_array(np.asarray(image_strip), height)
    w = W([_setup_array(glyph, height) for glyph in w.glyphs], w.codemap)

    w_ = torch.tensor(w.get_stacked_w())
    print(w_.min(), w_.max())
    print(image_strip.min(), image_strip.max())
    l_out = image_strip.shape[1]
    r = w_.shape[1]
    t = w_.shape[2]
    l_in = l_out - t + 1
    h_ = torch.rand(1, r, l_in)
    nmfd = nmf.NMFD(W=w_, H=h_, trainable_W=False, trainable_H=True)
    nmfd.fit(
        image_strip[None],
        max_iter=max_iter,
        beta=1,
        alpha=1,
        l1_ratio=0,
        verbose=True,
    )

    return nmfd, w.codemap
