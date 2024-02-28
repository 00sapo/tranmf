# cython: language_level=3
# distutils: language=c++
import cython as cy
import numpy as np
from tqdm import tqdm

# cdivision=True, boundscheck=False, wraparound=False, initializedcheck=False, -profile=True

DType = cy.fused_type(
    cy.short, cy.long, cy.longlong, cy.float, cy.double, cy.longdouble
)
Mat2D = cy.typedef(DType[:, :])


@cy.cclass
class LossComponent:
    @cy.ccall
    def majorize_h(self, row: int, col: int, h: Mat2D, w: Mat2D, v: Mat2D) -> float:
        return 0.0

    @cy.ccall
    def majorize_w(self, row: int, col: int, h: Mat2D, w: Mat2D, v: Mat2D) -> float:
        return 0.0

    @cy.ccall
    def minorize_h(self, row: int, col: int, h: Mat2D, w: Mat2D, v: Mat2D) -> float:
        return 0.0

    @cy.ccall
    def minorize_w(self, row: int, col: int, h: Mat2D, w: Mat2D, v: Mat2D) -> float:
        return 0.0

    @cy.ccall
    def compute(self, h: Mat2D, w: Mat2D, v: Mat2D, wh: Mat2D) -> float:
        return 0.0


@cy.cclass
class Euclidean2D(LossComponent):
    @cy.ccall
    def majorize_h(
        self, row: cy.Py_ssize_t, col: cy.Py_ssize_t, h: Mat2D, w: Mat2D, v: Mat2D
    ) -> float:
        return w[:, row] @ v[:, col]

    @cy.ccall
    def majorize_w(
        self, row: cy.Py_ssize_t, col: cy.Py_ssize_t, h: Mat2D, w: Mat2D, v: Mat2D
    ) -> float:
        return v[row, :] @ h[col, :]

    @cy.ccall
    def minorize_h(
        self, row: cy.Py_ssize_t, col: cy.Py_ssize_t, h: Mat2D, w: Mat2D, v: Mat2D
    ) -> float:
        sum: float = 0.0
        k: cy.Py_ssize_t
        f: cy.Py_ssize_t
        for k in range(h.shape[1]):
            h_sum = 0.0
            for f in range(h.shape[0]):
                h_sum += h[k, f] * h[col, f]
            sum += w[row, k] * h_sum
        return sum

    @cy.ccall
    def minorize_w(
        self, row: cy.Py_ssize_t, col: cy.Py_ssize_t, h: Mat2D, w: Mat2D, v: Mat2D
    ) -> float:
        sum: float = 0.0

        k: cy.Py_ssize_t
        f: cy.Py_ssize_t
        for k in range(w.shape[0]):
            w_sum = 0.0
            for f in range(w.shape[1]):
                w_sum += w[f, row] * w[f, k]
            sum += w_sum * h[k, col]
        return sum

    @cy.ccall
    def compute(self, h: Mat2D, w: Mat2D, v: Mat2D, wh: Mat2D) -> float:
        sum: float = 0.0
        i: cy.Py_ssize_t
        j: cy.Py_ssize_t
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                sum += (v[i, j] - wh[i, j]) ** 2
        return sum


@cy.cclass
class _Loss2D:
    def __init__(self, update_type: str, components: list[LossComponent]):
        self.update_type = update_type
        self.components = components

    @cy.ccall
    def compute(self, h: Mat2D, w: Mat2D, v: Mat2D, wh: Mat2D) -> float:
        sum = 0.0
        for c in self.components:
            sum += c.compute(h, w, v, wh)
        return sum

    @cy.ccall
    def update(self, h: Mat2D, w: Mat2D, v: Mat2D, fix_h: bool, fix_w: bool):
        """Updates h and w in place using the majorization and minorization of each loss-component"""
        if not fix_h:
            h_copy = h.copy()
        else:
            h_copy = h

        if not fix_w:
            w_copy = w.copy()
        else:
            w_copy = w

        i: cy.Py_ssize_t
        j: cy.Py_ssize_t
        for i in range(h.shape[0]):
            if not fix_h:
                # if needed, in order to generalize, we should iterate over all the
                # other dimensions of h and w and pass a list of indices to the
                # loss components
                for j in range(h.shape[1]):
                    # j is the column index
                    majorize = 0.0
                    minorize = 0.0
                    for c in self.components:
                        majorize += c.majorize_h(i, j, h_copy, w_copy, v)
                        minorize += c.minorize_h(i, j, h_copy, w_copy, v)
                    if self.update_type == "multiplicative":
                        h[i, j] *= majorize / minorize
                    elif self.update_type == "additive":
                        h[i, j] += majorize - minorize

            if not fix_w:
                for j in range(w.shape[0]):
                    # j is the row index
                    majorize = 0.0
                    minorize = 0.0
                    for c in self.components:
                        majorize += c.majorize_w(j, i, h_copy, w_copy, v)
                        minorize += c.minorize_w(j, i, h_copy, w_copy, v)
                    if self.update_type == "multiplicative":
                        w[j, i] *= majorize / minorize
                    elif self.update_type == "additive":
                        w[j, i] += majorize - minorize


class NMF:
    """
    The code should expose a class NMF2D that has 3 2D arrays (`h`, `w`, and `v`) and a
    Loss object as attributes. The type Loss contains a vector of LossComponent. Each
    LossComponent has methods `compute`, `majorize_h`, `majorize_w`, `minorize_h`, and
    `minorize_w`, receiving as arguments the matrices `h`, `w`, `v`, and the
    re-constructed `w @ h`. The Loss also has these methods that call the respective
    methods of the LossComponent objects for each element of the `w` and `h` matrices.
    The NMF2D class should have two methods `fit` and `step` that take a number of
    iterations (int), a tolerance (float) as arguments, and two booleans `fix_h` and
    `fix_w`.

    Use typing annotations for compiling via cython.
    """

    def __init__(
        self,
        h: np.ndarray,
        w: np.ndarray,
        v: np.ndarray,
        loss_components: list[LossComponent],
        update_type: str,
    ):
        assert (
            w.shape[1] == h.shape[0]
        ), "w and h must be compatible for matrix multiplication"
        assert w.shape[0] == v.shape[0], "w and v must have the same number of rows"
        assert h.shape[1] == v.shape[1], "h and v must have the same number of columns"

        self.w = w
        self.h = h
        self.v = v
        self.loss = _Loss2D(update_type, loss_components)

    def fit(self, n_iter: int, tol: float, fix_h: bool, fix_w: bool):
        """
        Run the NMF algorithm for n_iter iterations or until the relative change in the
        loss is less than tol. If fix_h is True, the algorithm should not update the h
        matrix. If fix_w is True, the algorithm should not update the w matrix.
        """
        if fix_h and fix_w:
            raise ValueError("At least one of fix_h and fix_w must be False")

        for _ in tqdm(range(n_iter)):
            new_loss = self.loss.compute(self.h, self.w, self.v, self.w @ self.h)
            if new_loss < tol:
                break
            self.loss.update(self.h, self.w, self.v, fix_h, fix_w)
