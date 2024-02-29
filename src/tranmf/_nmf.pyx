# cython: language_level=3
# distutils: language=c++, cdivision=True, boundscheck=False, wraparound=False, initializedcheck=False, -profile=True
from typing import Callable

cimport numpy as np
from tqdm import tqdm

np.import_array()


ctypedef np.float64_t DType
# DType = cy.fused_type(
#     np.float32_t,
#     np.float64_t,
#     np.int32_t,
#     np.int64_t,
#     np.int16_t,
#     np.int8_t,
# )
#

cdef class Euclidean2D:
    learning_rate_h: float
    learning_rate_w: float

    def __init__(self, learning_rate_h: float, learning_rate_w: float):
        self.learning_rate_h = learning_rate_h
        self.learning_rate_w = learning_rate_w

    cdef double majorize_h(
        self, int row, int col, np.ndarray[DType, ndim=2] h, np.ndarray[DType, ndim=2] w, np.ndarray[DType, ndim=2] v
    ):
        return self.learning_rate_h * (w[:, row] @ v[:, col])

    cdef double majorize_w(
        self, int row, int col, np.ndarray[DType, ndim=2] h, np.ndarray[DType, ndim=2] w, np.ndarray[DType, ndim=2] v
    ):
        return self.learning_rate_w * (v[row, :] @ h[col, :])

    cdef double minorize_h(
        self, int row, int col, np.ndarray[DType, ndim=2] h, np.ndarray[DType, ndim=2] w, np.ndarray[DType, ndim=2] v
    ):
        sum: float = 0.0

        # (W^T W H)_ij = \sum_k (W^T W)_ik H_kj =
        # = \sum_k (\sum_f (W^T_if W_fk) H_kj) =
        # \sum_k (\sum_f (W_fi W_fk) H_kj) =

        cdef int k
        for k in range(h.shape[0]):
            sum += (w[:, row] @ w[:, k]) * h[k, col]
        return sum * self.learning_rate_h

    cdef double minorize_w(
        self, int row, int col, np.ndarray[DType, ndim=2] h, np.ndarray[DType, ndim=2] w, np.ndarray[DType, ndim=2] v
    ):
        # (W H H^T)_ij = \sum_k (W_ik \sum_f(Hkf H^T_fj)_kj
        sum: float = 0.0
        cdef int k
        for k in range(h.shape[0]):
            sum += w[row, k] * (h[k, :] @ h[col, :])
        return sum * self.learning_rate_w

    cdef double compute(self, np.ndarray[DType, ndim=2] h, np.ndarray[DType, ndim=2] w, np.ndarray[DType, ndim=2] v, np.ndarray[DType, ndim=2] wh):
        sum: float = 0.0
        cdef int i, j
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                sum += (v[i, j] - wh[i, j]) ** 2
        return sum


cdef class _Loss2D:
    update_type: str
    components: list

    def __init__(self, update_type: str, components: list):
        self.update_type = update_type
        self.components = components

    cdef double compute(self, np.ndarray[DType, ndim=2] h, np.ndarray[DType, ndim=2] w, np.ndarray[DType, ndim=2] v, np.ndarray[DType, ndim=2] wh):
        sum = 0.0
        for c in self.components:
            if c.__class__ == Euclidean2D:
                c_ = <Euclidean2D> c
            else:
                raise ValueError(f"Unsupported loss component {c.__class__}")
            sum += c_.compute(h, w, v, wh)
        return sum

    cdef update(self, np.ndarray[DType, ndim=2] h, np.ndarray[DType, ndim=2] w, np.ndarray[DType, ndim=2] v, bint fix_h, bint fix_w):
        """Updates h and w in place using the majorization and minorization of each loss-component"""
        if not fix_h:
            h_copy = h.copy()
        else:
            h_copy = h

        if not fix_w:
            w_copy = w.copy()
        else:
            w_copy = w

        cdef int i, j
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
                        if c.__class__ == Euclidean2D:
                            c_ = <Euclidean2D> c
                        else:
                            raise ValueError(
                                f"Unsupported loss component {c.__class__}"
                            )
                        majorize += c_.majorize_h(i, j, h_copy, w_copy, v)
                        minorize += c_.minorize_h(i, j, h_copy, w_copy, v)
                    if self.update_type == "multiplicative":
                        h[i, j] = h[i, j] * majorize / minorize
                    elif self.update_type == "additive":
                        h[i, j] = h[i, j] * majorize - minorize

            if not fix_w:
                for j in range(w.shape[0]):
                    # j is the row index
                    majorize = 0.0
                    minorize = 0.0
                    for c in self.components:
                        if c.__class__ == Euclidean2D:
                            c_ = <Euclidean2D> c
                        else:
                            raise ValueError(
                                f"Unsupported loss component {c.__class__}"
                            )
                        majorize += c_.majorize_w(j, i, h_copy, w_copy, v)
                        minorize += c_.minorize_w(j, i, h_copy, w_copy, v)
                    if self.update_type == "multiplicative":
                        w[j, i] = w[j, i] * majorize / minorize
                    elif self.update_type == "additive":
                        w[j, i] = w[j, i] + majorize - minorize


cdef class NMF:
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

    Supported update types are "multiplicative" and "additive". If `alternate`
    is given, it should be a function that decides if swapping fix_h and fix_w
    at each iteration. The input is the current iteration number, and the
    output is a bool.
    """

    cdef _Loss2D loss
    cdef bint verbose
    cdef str update_type
    cdef object alternate  # without specific Callable typing, we default to 'object'
    cdef double _loss
    cdef int _iter

    def __init__(
        self,
        list loss_components,
        str update_type,
        alternate = None,
        bint verbose = False
    ):
        if alternate is None:
            alternate = lambda x: True

        self.loss = _Loss2D(update_type, loss_components)
        self.verbose = verbose
        self.update_type = update_type
        self.alternate = alternate

    cpdef fit(self,
             np.ndarray[DType, ndim=2] w,
             np.ndarray[DType, ndim=2] h,
             np.ndarray[DType, ndim=2] v, int n_iter, double tol, bint fix_h, bint fix_w):
        """
        Run the NMF algorithm for n_iter iterations or until the relative change in the
        loss is less than tol. If fix_h is True, the algorithm should not update the h
        matrix. If fix_w is True, the algorithm should not update the w matrix.

        Using `tol` is much slower.
        """
        assert (
            w.shape[1] == h.shape[0]
        ), "w and h must be compatible for matrix multiplication"
        assert w.shape[0] == v.shape[0], "w and v must have the same number of rows"
        assert h.shape[1] == v.shape[1], "h and v must have the same number of columns"
        if not fix_h and not fix_w:
            raise ValueError("At least one of fix_h and fix_w must be False")

        if self.verbose:
            iterator = range(n_iter)
        else:
            iterator = tqdm(range(n_iter))
        self._loss = 0.0
        for self._iter in iterator:
            if self.alternate is not None:
                if self.alternate(self._iter):
                    fix_h = not fix_h
                    fix_w = not fix_w
            if tol > 0:
                self._loss = self.loss.compute(h, w, v, w @ h)
                if self._loss < tol:
                    break

            self.loss.update(h, w, v, fix_h, fix_w)
            if self.verbose:
                print(f"iter {self._iter}: loss={self._loss}")