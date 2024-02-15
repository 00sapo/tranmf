import shutil
import subprocess
import tempfile
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Union

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
