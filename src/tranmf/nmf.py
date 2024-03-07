import shutil
import subprocess
import tempfile
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.signal import convolve
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

THIS_DIR = Path(__file__).parent


@dataclass
class W:
    #: the W matrix (H x glyphs)
    glyphs: list[np.ndarray]

    #: the list of hex codes and its index position
    codemap: dict[str, int]

    def resize(self, height: int):
        """Resize the glyphs to match the given height."""
        for i, glyph in enumerate(self.glyphs):
            self.glyphs[i] = cv2.resize(
                glyph, (round(glyph.shape[1] * height / glyph.shape[0]), height)
            )

    def get_concatenated_w(self):
        """Create a new W matrix from a list of arrays and a codemap."""
        accumulated_codemap = {}
        start = 0
        for k, v in self.codemap.items():
            accumulated_codemap[k] = start, start + self.glyphs[v].shape[1]
            start += self.glyphs[v].shape[1]

        return np.concatenate(self.glyphs, axis=1), accumulated_codemap

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
        image_array = np.asarray(Image.open(outfile).convert("L"))
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
    arr = jax.device_put(arr)
    return arr


def build_network(params):
    assert params["n_conv_layers"] == len(params["activations"])

    # @jax.jit
    def network(w, h, weights):
        assert params["n_conv_layers"] == len(weights) - 1
        # W
        w = get_transformed_w(w)

        # H
        for i in range(params["n_conv_layers"]):
            h = convolve(h, weights[i], mode="same")
            h = params["activations"][i](h)
            # h = jax.lax.max_pool(h, (1, kernel_size), (1, kernel_size), "SAME")
        h = h * weights[-1]
        h = h.mean(axis=2)
        h = get_transformed_h(h)
        return jnp.dot(w, h), w, h

    return network


# @jax.jit
def get_transformed_w(w_in):
    # keep positive
    w_in = jax.nn.relu(w_in)
    return w_in


# @jax.jit
def get_transformed_h(h_in):
    # one-hot encoding
    h_in = jax.nn.softmax(h_in, axis=0)
    # keep positive
    h_in = jax.nn.relu(h_in)
    return h_in


# @jax.jit
def structural_contiguity_loss(h):
    loss = jnp.zeros_like(h)
    k = 4
    for k_ in range(1, k + 1):
        loss = loss.at[:-k_, :-k_].set(jnp.abs(h[:-k_, :-k_] - h[k_:, k_:]))
    return loss.sum() / k


# @jax.jit
def structural_separation_loss(h):
    k = 14
    h = h[:, : int(h.shape[1] / k) * k]
    h = h.reshape(h.shape[0], k, -1)
    h = h.sum(axis=2)
    m = h.size
    return m - jnp.abs(h[:, :-1] - h[:, 1:]).sum()


# @jax.jit
def l1_loss(wh, v):
    return jnp.abs(wh - v).sum()


# @jax.jit
def nmf_loss(w, h_weights, v, network):
    h, weights = h_weights
    wh, w, h = network(w, h, weights)
    # loss_h = (jnp.std(h, axis=1)).sum()
    # loss_h = jnp.sum(jnp.abs(1 - jnp.max(h, axis=0)))
    return (
        l1_loss(wh, v)
        # + loss_h
        # + structural_contiguity_loss(h)
        # + structural_separation_loss(h)
    )


def get_alternating_gradient(loss_fn, idx, w_shape, h_shape, weights):
    def adjust_gradient(*args, **kwargs):
        val, grad = jax.value_and_grad(loss_fn, idx)(*args, **kwargs)
        if idx == (0,):
            grad = grad[0], (jnp.zeros(h_shape), (jnp.zeros(w.shape) for w in weights))
        elif idx == (1,):
            grad = jnp.zeros(w_shape), grad[0]
        return val, grad

    return adjust_gradient


def init_weights(params):
    rng_key = jax.random.PRNGKey(1997)
    weights = [
        jax.random.normal(rng_key, params["kernel_size"])
        for _ in range(params["n_conv_layers"])
    ]
    weights.append(jnp.ones((1, 1, params["h_channel_size"])))
    return weights


def _iterate(
    w_,
    h,
    image_strip,
    network,
    network_params,
    optimizer,
    max_iter,
    alternate,
    freeze_h,
    freeze_w,
    tol,
    patience,
    verbose,
):
    weights = init_weights(network_params)
    opt_state = optimizer.init((w_, (h, weights)))

    def get_alternating_gradient_(idx):
        return get_alternating_gradient(nmf_loss, idx, w_.shape, h.shape, weights)

    best_wh = (w_, h, weights)
    best_loss = jnp.inf
    last_improvement_iter = -1
    for i in range(max_iter):
        # decide what to freeze
        if alternate is not None and alternate(i):
            freeze_w = not freeze_w
            freeze_h = not freeze_h
        if freeze_h:
            nmf_loss_ = get_alternating_gradient_((0,))
        elif freeze_w:
            nmf_loss_ = get_alternating_gradient_((1,))
        else:
            nmf_loss_ = get_alternating_gradient_((0, 1))

        # compute the loss and gradients
        loss_value, grads = nmf_loss_(w_, (h, weights), image_strip, network)
        if verbose:
            print(f"Loss at iteration {i}: {loss_value}")
        updates, opt_state = optimizer.update(grads, opt_state)
        w_, (h, weights) = optax.apply_updates((w_, (h, weights)), updates)

        # early stopping
        if loss_value < best_loss:
            best_wh = (w_, h, weights)
            last_improvement_iter = i
            if best_loss - loss_value < tol:
                print("Converged.")
                break
            else:
                best_loss = loss_value
        else:
            if i - last_improvement_iter > patience:
                print("Early stopping.")
                break

    return best_wh


def run_single_nmf(
    image_strip: np.ndarray,
    w: W,
    *,
    max_iter=50,
    height=50,
    optimizer=optax.adam,
    learning_rate=0.1,
    tol=1e-5,
    freeze_w=False,
    freeze_h=False,
    alternate: Optional[
        Callable[
            [
                int,
            ],
            bool,
        ]
    ] = None,
    verbose=False,
    patience=100,
    network_params={
        "n_conv_layers": 2,
        "h_channel_size": 10,
        "kernel_size": (1, 15, 10),
        "activations": [jax.nn.relu, jax.nn.relu],
    },
) -> tuple[np.ndarray, np.ndarray, dict[str, int], np.ndarray]:
    """Run NMF on a single image strip.
    Args:
        image_strip (np.ndarray): The image strip to run NMF on.
        W (W): The W matrix.
        max_iter (int, optional): Maximum number of iterations. Defaults to 50.
        height (int, optional): The height used for the W and final WH matrix. Defaults to 50.
        oprtimizer (optax.GradientTransformation, optional): The optimizer to use. Defaults to optax.adam.
        ... (other args)
        alternate (Callable[int, bool], optional): A function that returns True if the
            iteration should alternate between updating W and H. Defaults to None.
    Returns:
        np.ndarray: The W matrix.
        np.ndarray: The H matrix.
        dict[str, int]: The codemap.
    """
    assert not (freeze_w and freeze_h), "Cannot freeze both W and H."

    # put image in grayscale mode
    # resize image strip to match W's height
    # convert to float64 array in 0-1
    image_strip = _setup_array(np.asarray(image_strip), height)
    w.resize(height)
    w_, codemap = w.get_concatenated_w()
    w_ = _setup_array(w_, height)

    # random h
    # key = jax.random.PRNGKey(1997)
    # h = jax.random.uniform(key, (w_.shape[1], image_strip.shape[1]))
    h = jnp.zeros(
        (w_.shape[1], image_strip.shape[1], network_params["h_channel_size"]),
        dtype=jnp.float32,
    )
    # but give more a chance to each glyph to start in any position
    for start, end in codemap.values():
        h = h.at[start].set(1.0)
    network = build_network(network_params)
    best_wh = _iterate(
        w_,
        h,
        image_strip,
        network,
        network_params,
        optimizer(learning_rate),
        max_iter,
        alternate,
        freeze_h,
        freeze_w,
        tol,
        patience,
        verbose,
    )

    # TODO: w_ should be used to change the glyphs of w
    # w_ = w_.reshape(w.glyphs[0].shape[0], w.glyphs[0].shape[1], -1)
    # w.glyphs = [w_[:, :, i] for i in range(w_.shape[2])]
    wh, w_, h = network(*best_wh)
    return np.asarray(w_), np.asarray(h), codemap, np.asarray(wh)
