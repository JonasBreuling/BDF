import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse, csc_matrix, eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from scipy.integrate._ivp.common import (
    validate_max_step,
    validate_tol,
    select_initial_step,
    norm,
    EPS,
    num_jac,
    validate_first_step,
    warn_extraneous,
)

from scipy.integrate._ivp.base import DenseOutput, ConstantDenseOutput
from scipy.integrate._ivp.ivp import prepare_events, MESSAGES, OdeResult

# from cardillo.math import fsolve


# source code taken from https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/bdf.py
MAX_ORDER = 5
NEWTON_MAXITER = 4
# MIN_FACTOR = 0.2
# MAX_FACTOR = 10
# NEWTON_MAXITER = 1
# MIN_FACTOR = 1.0
# MAX_FACTOR = 1.0
# TODO: the lower boundary is crucial
MIN_FACTOR = 0.999
MAX_FACTOR = 10


def compute_R(order, factor):
    """Compute the matrix for changing the differences array."""
    I = np.arange(1, order + 1)[:, None]
    J = np.arange(1, order + 1)
    M = np.zeros((order + 1, order + 1))
    M[1:, 1:] = (I - 1 - factor * J) / I
    M[0] = 1
    return np.cumprod(M, axis=0)


def change_D(D, order, factor):
    """Change differences array in-place when step size is changed."""
    R = compute_R(order, factor)
    U = compute_R(order, 1)
    RU = R.dot(U)
    D[: order + 1] = np.dot(RU.T, D[: order + 1])


def solve_bdf_system(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol):
    """Solve the algebraic system resulting from BDF method."""
    d = 0
    y = y_predict.copy()
    dy_norm_old = None
    converged = False
    for k in range(NEWTON_MAXITER):
        y_dot = psi + c * d
        # f = fun(t_new, y)
        f = fun(t_new, y, y_dot)
        if not np.all(np.isfinite(f)):
            break

        # dy = solve_lu(LU, c * f - psi - d)
        dy = solve_lu(LU, -f)
        dy_norm = norm(dy / scale)

        if dy_norm_old is None:
            rate = None
        else:
            rate = dy_norm / dy_norm_old

        if rate is not None and (
            rate >= 1 or rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol
        ):
            break

        y += dy
        d += dy

        if dy_norm == 0 or rate is not None and rate / (1 - rate) * dy_norm < tol:
            converged = True
            break

        dy_norm_old = dy_norm

    return converged, k + 1, y, d


def check_arguments_DAE(fun, y0, y_dot0, support_complex):
    """Helper function for checking arguments common to all solvers."""
    y0 = np.asarray(y0)
    y_dot0 = np.asarray(y_dot0)
    if np.issubdtype(y0.dtype, np.complexfloating) or np.issubdtype(
        y_dot0.dtype, np.complexfloating
    ):
        if not support_complex:
            raise ValueError(
                "`y0` or `y_dot0`is complex, but the chosen solver does "
                "not support integration in a complex domain."
            )
        dtype = complex
    else:
        dtype = float
    y0 = y0.astype(dtype, copy=False)
    y_dot0 = y_dot0.astype(dtype, copy=False)

    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")
    if y_dot0.ndim != 1:
        raise ValueError("`y_dot0` must be 1-dimensional.")

    if not np.isfinite(y0).all():
        raise ValueError("All components of the initial state `y0` must be finite.")
    if not np.isfinite(y_dot0).all():
        raise ValueError("All components of the initial state `y_dot0` must be finite.")

    def fun_wrapped(t, y, y_dot):
        return np.asarray(fun(t, y, y_dot), dtype=dtype)

    return fun_wrapped, y0, y_dot0


class DAESolver:
    """Base class for DAE solvers.

    TODO: add new documentation.

    In order to implement a new solver you need to follow the guidelines:

        1. A constructor must accept parameters presented in the base class
           (listed below) along with any other parameters specific to a solver.
        2. A constructor must accept arbitrary extraneous arguments
           ``**extraneous``, but warn that these arguments are irrelevant
           using `common.warn_extraneous` function. Do not pass these
           arguments to the base class.
        3. A solver must implement a private method `_step_impl(self)` which
           propagates a solver one step further. It must return tuple
           ``(success, message)``, where ``success`` is a boolean indicating
           whether a step was successful, and ``message`` is a string
           containing description of a failure if a step failed or None
           otherwise.
        4. A solver must implement a private method `_dense_output_impl(self)`,
           which returns a `DenseOutput` object covering the last successful
           step.
        5. A solver must have attributes listed below in Attributes section.
           Note that ``t_old`` and ``step_size`` are updated automatically.
        6. Use `fun(self, t, y)` method for the system rhs evaluation, this
           way the number of function evaluations (`nfev`) will be tracked
           automatically.
        7. For convenience, a base class provides `fun_single(self, t, y)` and
           `fun_vectorized(self, t, y)` for evaluating the rhs in
           non-vectorized and vectorized fashions respectively (regardless of
           how `fun` from the constructor is implemented). These calls don't
           increment `nfev`.
        8. If a solver uses a Jacobian matrix and LU decompositions, it should
           track the number of Jacobian evaluations (`njev`) and the number of
           LU decompositions (`nlu`).
        9. By convention, the function evaluations used to compute a finite
           difference approximation of the Jacobian should not be counted in
           `nfev`, thus use `fun_single(self, t, y)` or
           `fun_vectorized(self, t, y)` when computing a finite difference
           approximation of the Jacobian.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
        return an array of the same shape as ``y``. See `vectorized` for more
        information.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time --- the integration won't continue beyond it. It also
        determines the direction of the integration.
    vectorized : bool
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by methods 'Radau' and 'BDF', but
        will result in slower execution for other methods. It can also
        result in slower overall execution for 'Radau' and 'BDF' in some
        circumstances (e.g. small ``len(y0)``).
    support_complex : bool, optional
        Whether integration in a complex domain should be supported.
        Generally determined by a derived solver class capabilities.
        Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of the system's rhs evaluations.
    njev : int
        Number of the Jacobian evaluations.
    nlu : int
        Number of LU decompositions.
    """

    TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

    def __init__(self, fun, t0, y0, y_dot0, t_bound, vectorized, support_complex=False):
        self.t_old = None
        self.t = t0
        self._fun, self.y, self.y_dot = check_arguments_DAE(
            fun, y0, y_dot0, support_complex
        )
        self.t_bound = t_bound
        self.vectorized = vectorized

        if vectorized:

            def fun_single(t, y, y_dot):
                return self._fun(t, y[:, None], y_dot[:, None]).ravel()

            fun_vectorized = self._fun
        else:
            fun_single = self._fun

            def fun_vectorized(t, y, y_dot):
                raise RuntimeError("This has to be checked!")
                f = np.empty_like(y)
                for i, (yi, y_doti) in enumerate(zip(y.T, y_dot.T)):
                    f[:, i] = self._fun(t, yi, y_doti)
                return f

        def fun(t, y, y_dot):
            self.nfev += 1
            return self.fun_single(t, y, y_dot)

        self.fun = fun
        self.fun_single = fun_single
        self.fun_vectorized = fun_vectorized

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.n = self.y.size
        assert self.y.size == self.y_dot.size
        self.status = "running"

        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    @property
    def step_size(self):
        if self.t_old is None:
            return None
        else:
            return np.abs(self.t - self.t_old)

    def step(self):
        """Perform one integration step.

        Returns
        -------
        message : string or None
            Report from the solver. Typically a reason for a failure if
            `self.status` is 'failed' after the step was taken or None
            otherwise.
        """
        if self.status != "running":
            raise RuntimeError("Attempt to step on a failed or finished " "solver.")

        if self.n == 0 or self.t == self.t_bound:
            # Handle corner cases of empty solver or no integration.
            self.t_old = self.t
            self.t = self.t_bound
            message = None
            self.status = "finished"
        else:
            t = self.t
            success, message = self._step_impl()

            if not success:
                self.status = "failed"
            else:
                self.t_old = t
                if self.direction * (self.t - self.t_bound) >= 0:
                    self.status = "finished"

        return message

    def dense_output(self):
        """Compute a local interpolant over the last successful step.

        Returns
        -------
        sol : `DenseOutput`
            Local interpolant over the last successful step.
        """
        if self.t_old is None:
            raise RuntimeError(
                "Dense output is available after a successful " "step was made."
            )

        if self.n == 0 or self.t == self.t_old:
            # Handle corner cases of empty solver and no integration.
            return ConstantDenseOutput(self.t_old, self.t, self.y)
        else:
            return self._dense_output_impl()

    def _step_impl(self):
        raise NotImplementedError

    def _dense_output_impl(self):
        raise NotImplementedError


class BDF(DAESolver):
    """Implicit method based on backward-differentiation formulas.

    This is a variable order method with the order varying automatically from
    1 to 5. The general framework of the BDF algorithm is described in [1]_.
    This class implements a quasi-constant step size as explained in [2]_.
    The error estimation strategy for the constant-step BDF is derived in [3]_.
    An accuracy enhancement using modified formulas (NDF) [2]_ is also implemented.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
        return an array of the same shape as ``y``. See `vectorized` for more
        information.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : {None, array_like, sparse_matrix, callable}, optional
        Jacobian matrix of the right-hand side of the system with respect to y,
        required by this method. The Jacobian matrix has shape (n, n) and its
        element (i, j) is equal to ``d f_i / d y_j``.
        There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)`` as necessary.
              For the 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
        elements in *each* row, providing the sparsity structure will greatly
        speed up the computations [4]_. A zero entry means that a corresponding
        element in the Jacobian is always zero. If None (default), the Jacobian
        is assumed to be dense.
    vectorized : bool, optional
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by this method, but may result in slower
        execution overall in some circumstances (e.g. small ``len(y0)``).

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.

    References
    ----------
    .. [1] G. D. Byrne, A. C. Hindmarsh, "A Polyalgorithm for the Numerical
           Solution of Ordinary Differential Equations", ACM Transactions on
           Mathematical Software, Vol. 1, No. 1, pp. 71-96, March 1975.
    .. [2] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
    .. [3] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations I:
           Nonstiff Problems", Sec. III.2.
    .. [4] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13, pp. 117-120, 1974.
    """

    def __init__(
        self,
        fun,
        t0,
        y0,
        y_dot0,
        t_bound,
        max_step=np.inf,
        rtol=1e-3,
        atol=1e-6,
        jac=None,
        jac_sparsity=None,
        vectorized=False,
        first_step=None,
        **extraneous,
    ):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, y_dot0, t_bound, vectorized, support_complex=True)

        # # solve for initial accelerations
        # f = lambda y_dot: self.fun(t0, y0, y_dot)
        # y_dot0, converged, error, n_iter, F = fsolve(f, y_dot0)
        # assert converged, "Failed to solve for consistent initial conditions."
        # self.y_dot = y_dot0

        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        f = self.fun(self.t, self.y, self.y_dot)
        # TODO: Implement a starting routine here!
        # if first_step is None:
        #     self.h_abs = select_initial_step(
        #         self.fun, self.t, self.y, f, self.direction, 1, self.rtol, self.atol
        #     )
        # else:
        #     self.h_abs = validate_first_step(first_step, t0, t_bound)
        assert first_step is not None, "initial step size has to be given"
        self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.h_abs_old = None
        self.error_norm_old = None

        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol**0.5))

        kappa = np.array([0, -0.1850, -1 / 9, -0.0823, -0.0415, 0])[: MAX_ORDER + 1]
        self.gamma = np.hstack(
            (0, np.cumsum(1 / np.arange(1, MAX_ORDER + 1)))
        )  # factor of leading coefficient
        self.alpha = (1 - kappa) * self.gamma  # 1 x L-vector of Byrne1975 (2.18)
        self.error_const = kappa * self.gamma + 1 / np.arange(1, MAX_ORDER + 2)

        D = np.empty((MAX_ORDER + 3, self.n), dtype=self.y.dtype)
        D[0] = self.y
        # D[1] = f * self.h_abs * self.direction
        D[1] = self.y_dot * self.h_abs * self.direction
        self.D = D

        self.order = 1
        self.n_equal_steps = 0
        self.LU = None

        self.jac_factor = None
        self.jac, self.J = self._validate_jac(jac, jac_sparsity)
        if issparse(self.J):

            def lu(A):
                self.nlu += 1
                return splu(A)

            def solve_lu(LU, b):
                return LU.solve(b)

            I = eye(self.n, format="csc", dtype=self.y.dtype)
        else:

            def lu(A):
                self.nlu += 1
                return lu_factor(A, overwrite_a=True)

            def solve_lu(LU, b):
                return lu_solve(LU, b, overwrite_b=True)

            I = np.identity(self.n, dtype=self.y.dtype)

        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I

    def _validate_jac(self, jac, sparsity):
        t0 = self.t
        y0 = self.y
        y_dot0 = self.y_dot
        c = self.h_abs * self.direction / self.alpha[self.order]

        if jac is None:
            raise RuntimeError("Jacobian has to be given for the moment")
            if sparsity is not None:
                if issparse(sparsity):
                    sparsity = csc_matrix(sparsity)
                groups = group_columns(sparsity)
                sparsity = (sparsity, groups)

            def jac_wrapped(t, y):
                self.njev += 1
                f = self.fun_single(t, y)
                J, self.jac_factor = num_jac(
                    self.fun_vectorized, t, y, f, self.atol, self.jac_factor, sparsity
                )
                return J

            J = jac_wrapped(t0, y0)
        elif callable(jac):
            J = jac(t0, y0, y_dot0, c)
            self.njev += 1
            if issparse(J):
                J = csc_matrix(J, dtype=np.common_type(y0, y_dot0))

                def jac_wrapped(t, y, y_dot, c):
                    self.njev += 1
                    return csc_matrix(
                        jac(t, y, y_dot, c), dtype=np.common_type(y0, y_dot0)
                    )

            else:
                J = np.asarray(J, dtype=np.common_type(y0, y_dot0))

                def jac_wrapped(t, y, y_dot, c):
                    self.njev += 1
                    return np.asarray(
                        jac(t, y, y_dot, c), dtype=np.common_type(y0, y_dot0)
                    )

            if J.shape != (self.n, self.n):
                raise ValueError(
                    "`jac` is expected to have shape {}, but "
                    "actually has {}.".format((self.n, self.n), J.shape)
                )
        else:
            if issparse(jac):
                J = csc_matrix(jac, dtype=np.common_type(y0, y_dot0))
            else:
                J = np.asarray(jac, dtype=np.common_type(y0, y_dot0))

            if J.shape != (self.n, self.n):
                raise ValueError(
                    "`jac` is expected to have shape {}, but "
                    "actually has {}.".format((self.n, self.n), J.shape)
                )
            jac_wrapped = None

        return jac_wrapped, J

    def _step_impl(self):
        t = self.t
        D = self.D

        max_step = self.max_step
        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            change_D(D, self.order, max_step / self.h_abs)
            self.n_equal_steps = 0
        elif self.h_abs < min_step:
            h_abs = min_step
            change_D(D, self.order, min_step / self.h_abs)
            self.n_equal_steps = 0
        else:
            h_abs = self.h_abs

        # min_step = max_step
        # h_abs = max_step

        atol = self.atol
        rtol = self.rtol
        order = self.order

        alpha = self.alpha
        gamma = self.gamma
        error_const = self.error_const

        J = self.J
        LU = self.LU
        current_jac = self.jac is None

        step_accepted = False
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
                change_D(D, order, np.abs(t_new - t) / h_abs)
                self.n_equal_steps = 0
                LU = None

            h = t_new - t
            h_abs = np.abs(h)

            y_predict = np.sum(D[: order + 1], axis=0)
            y_dot_predict = np.dot(D[1 : order + 1].T, gamma[1 : order + 1]) / h

            scale = atol + rtol * np.abs(y_predict)
            # psi = np.dot(D[1 : order + 1].T, gamma[1 : order + 1]) / alpha[order]

            # def f(d):
            #     y = y_predict + d
            #     y_dot = (
            #         gamma[order] * d + np.dot(D[1 : order + 1].T, gamma[1 : order + 1])
            #     ) / h
            #     return self.fun(t_new, y, y_dot)

            # d, converged, error, n_iter, F = fsolve(f, np.zeros_like(y_predict))
            # y_new = y_predict + d

            # def f(y):
            #     d = y - y_predict.copy()
            #     y_dot = (
            #         gamma[order] * d + np.dot(D[1 : order + 1].T, gamma[1 : order + 1])
            #     ) / h
            #     return self.fun(t_new, y, y_dot)

            # y_new, converged, error, n_iter, F = fsolve(f, y_predict.copy())
            # d = y_new - y_predict

            # # print(f"t: {t_new}; converged: {converged}")
            print(f"t: {t_new}")

            converged = False
            # c = h / alpha[order]
            c = gamma[order] / h
            while not converged:
                if LU is None:
                    # TODO: LU-decomposition
                    # LU = self.lu(self.I - c * J)
                    LU = self.lu(J)

                converged, n_iter, y_new, d = solve_bdf_system(
                    self.fun,
                    t_new,
                    y_predict,
                    c,
                    # psi,
                    y_dot_predict,
                    LU,
                    self.solve_lu,
                    scale,
                    self.newton_tol,
                )

                if not converged:
                    if current_jac:
                        break
                    # J = self.jac(t_new, y_predict)
                    J = self.jac(t_new, y_predict, y_dot_predict, c)
                    # J = self.jac(t_new, y_predict + d, y_dot_predict + c * d, c)
                    LU = None
                    current_jac = True

            # accept all steps
            step_accepted = True

            if not converged:
                # print(f"not converged")
                # factor = 0.5
                # TODO: This factor is a problem
                factor = 0.6
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                LU = None
                continue

            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

            scale = atol + rtol * np.abs(y_new)
            error = error_const[order] * d
            # error = error_const[order] * F # TODO: This seems to fix the problem
            # d_ = np.zeros_like(d)
            # d_[:4] = d[:4]
            # error = error_const[order] * d_

            # supress algebraic variables in the error measure
            error[-1] = 0
            scale[-1] = 1

            error_norm = norm(error / scale)

            if error_norm > 1:
                factor = max(MIN_FACTOR, safety * error_norm ** (-1 / (order + 1)))
                # factor = 1
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                # As we didn't have problems with convergence, we don't
                # reset LU here.
            else:
                step_accepted = True

        self.n_equal_steps += 1

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.J = J
        self.LU = LU

        # Update differences. The principal relation here is
        # D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}. Keep in mind that D
        # contained difference for previous interpolating polynomial and
        # d = D^{k + 1} y_n. Thus this elegant code follows.
        D[order + 2] = d - D[order + 1]
        D[order + 1] = d
        for i in reversed(range(order + 1)):
            D[i] += D[i + 1]

        if self.n_equal_steps < order + 1:
            return True, None

        if order > 1:
            error_m = error_const[order - 1] * D[order]
            error_m_norm = norm(error_m / scale)
        else:
            error_m_norm = np.inf

        if order < MAX_ORDER:
            error_p = error_const[order + 1] * D[order + 2]
            error_p_norm = norm(error_p / scale)
        else:
            error_p_norm = np.inf

        error_norms = np.array([error_m_norm, error_norm, error_p_norm])
        with np.errstate(divide="ignore"):
            factors = error_norms ** (-1 / np.arange(order, order + 3))

        delta_order = np.argmax(factors) - 1
        order += delta_order
        self.order = order

        factor = min(MAX_FACTOR, safety * np.max(factors))
        self.h_abs *= factor
        change_D(D, order, factor)
        self.n_equal_steps = 0
        self.LU = None

        return True, None

    def _dense_output_impl(self):
        return BdfDenseOutput(
            self.t_old,
            self.t,
            self.h_abs * self.direction,
            self.order,
            self.D[: self.order + 1].copy(),
        )


class BdfDenseOutput(DenseOutput):
    def __init__(self, t_old, t, h, order, D):
        super().__init__(t_old, t)
        self.order = order
        self.t_shift = self.t - h * np.arange(self.order)
        self.denom = h * (1 + np.arange(self.order))
        self.D = D

    def _call_impl(self, t):
        if t.ndim == 0:
            x = (t - self.t_shift) / self.denom
            p = np.cumprod(x)
        else:
            x = (t - self.t_shift[:, None]) / self.denom[:, None]
            p = np.cumprod(x, axis=0)

        y = np.dot(self.D[1:].T, p)
        if y.ndim == 1:
            y += self.D[0]
        else:
            y += self.D[0, :, None]

        return y


def solve_dae(
    fun,
    t_span,
    y0,
    y_dot0,
    t_eval=None,
    dense_output=False,
    events=None,
    vectorized=False,
    args=None,
    **options,
):
    """Solve an initial value problem for a system of ODEs.

    This function numerically integrates a system of implicit differential
    algebraic equations::

        0 = f(t, y, y_dot)
        y(t0) = y0
        y_dot(t0) = y_dot0

    Here t is a 1-D independent variable (time), y(t) is an
    N-D vector-valued function (state), y_dot(t) is an N-D vector-valued
    function (state derivatives), and an N-D vector-valued function
    f(t, y, y_dot) determines the implicit differential algebraic equations.
    The goal is to find y(t) and y_dot(t) approximately satisfying the
    differential algebraic equations, given initial values y(t0)=y0 and
    y_dot(t0) = y_dot0.

    The solver uses an implicit multi-step variable-order (1 to 5) method
    based on a backward differentiation formula for the derivative
    approximation [5]_. The implementation follows the one described in [6]_.
    A quasi-constant step scheme is used and accuracy is enhanced using the
    NDF modification. Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Implicit function depending on time ``t``, the state ``y`` and the
        state derivative ``y_dot``. The calling signature is
        ``fun(t, y, y_dot)``, where ``t`` is a scalar and ``y``/ ``y_dot`` are
        ndarrays with ``len(y) = len(y0) = len(y_dot) ) len(y_dot0)``.
        Additional arguments need to be passed if ``args`` is used (see
        documentation of ``args`` argument). ``fun`` must return an array of
        the same shape as ``y``/ `ỳ_dot``. See `vectorized` for more information.
    t_span : 2-member sequence
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. Both t0 and tf must be floats
        or values interpretable by the float conversion function.
    y0 : array_like, shape (n,)
        Initial state. For problems in the complex domain, pass `y0` with a
        complex data type (even if the initial value is purely real).
    y_dot0 : array_like, shape (n,)
        Initial state derivative. For problems in the complex domain, pass
        `y_dot0` with a complex data type (even if the initial value is
        purely real).
    t_eval : array_like or None, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points selected by the solver.
    dense_output : bool, optional
        Whether to compute a continuous solution. Default is False.
    events : callable, or list of callables, optional
        Events to track. If None (default), no events will be tracked.
        Each event occurs at the zeros of a continuous function of time and
        state. Each function must have the signature ``event(t, y)`` where
        additional argument have to be passed if ``args`` is used (see
        documentation of ``args`` argument). Each function must return a
        float. The solver will find an accurate value of `t` at which
        ``event(t, y(t)) = 0`` using a root-finding algorithm. By default,
        all zeros will be found. The solver looks for a sign change over
        each step, so if multiple zero crossings occur within one step,
        events may be missed. Additionally each `event` function might
        have the following attributes:

            terminal: bool, optional
                Whether to terminate integration if this event occurs.
                Implicitly False if not assigned.
            direction: float, optional
                Direction of a zero crossing. If `direction` is positive,
                `event` will only trigger when going from negative to positive,
                and vice versa if `direction` is negative. If 0, then either
                direction will trigger event. Implicitly 0 if not assigned.

        You can assign attributes like ``event.terminal = True`` to any
        function in Python.
    vectorized : bool, optional
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by methods 'Radau' and 'BDF', but
        will result in slower execution for other methods and for 'Radau' and
        'BDF' in some circumstances (e.g. small ``len(y0)``).
    args : tuple, optional
        Additional arguments to pass to the user-defined functions.  If given,
        the additional arguments are passed to all user-defined functions.
        So if, for example, `fun` has the signature ``fun(t, y, a, b, c)``,
        then `jac` (if given) and any event functions must have the same
        signature, and `args` must be a tuple of length 3.
    **options
        Options passed to a chosen solver. All options available for already
        implemented solvers are listed below.
    first_step : float or None, optional
        Initial step size. Default is `None` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float or array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : array_like, sparse_matrix, callable or None, optional
        Jacobian matrix of the right-hand side of the system with respect
        to y, required by the 'Radau', 'BDF' and 'LSODA' method. The
        Jacobian matrix has shape (n, n) and its element (i, j) is equal to
        ``d f_i / d y_j``.  There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant. Not supported by 'LSODA'.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)``, as necessary.
              Additional arguments have to be passed if ``args`` is
              used (see documentation of ``args`` argument).
              For 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    jac_sparsity : array_like, sparse matrix or None, optional
        Defines a sparsity structure of the Jacobian matrix for a finite-
        difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few
        non-zero elements in *each* row, providing the sparsity structure
        will greatly speed up the computations [10]_. A zero entry means that
        a corresponding element in the Jacobian is always zero. If None
        (default), the Jacobian is assumed to be dense.
        Not supported by 'LSODA', see `lband` and `uband` instead.
    lband, uband : int or None, optional
        Parameters defining the bandwidth of the Jacobian for the 'LSODA'
        method, i.e., ``jac[i, j] != 0 only for i - lband <= j <= i + uband``.
        Default is None. Setting these requires your jac routine to return the
        Jacobian in the packed format: the returned array must have ``n``
        columns and ``uband + lband + 1`` rows in which Jacobian diagonals are
        written. Specifically ``jac_packed[uband + i - j , j] = jac[i, j]``.
        The same format is used in `scipy.linalg.solve_banded` (check for an
        illustration).  These parameters can be also used with ``jac=None`` to
        reduce the number of Jacobian elements estimated by finite differences.
    min_step : float, optional
        The minimum allowed step size for 'LSODA' method.
        By default `min_step` is zero.

    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Values of the solution at `t`.
    sol : `OdeSolution` or None
        Found solution as `OdeSolution` instance; None if `dense_output` was
        set to False.
    t_events : list of ndarray or None
        Contains for each event type a list of arrays at which an event of
        that type event was detected. None if `events` was None.
    y_events : list of ndarray or None
        For each value of `t_events`, the corresponding value of the solution.
        None if `events` was None.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.
    status : int
        Reason for algorithm termination:

            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `tspan`.
            *  1: A termination event occurred.

    message : string
        Human-readable description of the termination reason.
    success : bool
        True if the solver reached the interval end or a termination event
        occurred (``status >= 0``).

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [5] `Backward Differentiation Formula
            <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_
            on Wikipedia.
    .. [6] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
    .. [7] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
           Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
           pp. 55-64, 1983.
    .. [8] L. Petzold, "Automatic selection of methods for solving stiff and
           nonstiff systems of ordinary differential equations", SIAM Journal
           on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
           1983.
    .. [9] `Stiff equation <https://en.wikipedia.org/wiki/Stiff_equation>`_ on
           Wikipedia.
    .. [10] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
            sparse Jacobian matrices", Journal of the Institute of Mathematics
            and its Applications, 13, pp. 117-120, 1974.
    .. [11] `Cauchy-Riemann equations
             <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on
             Wikipedia.
    .. [12] `Lotka-Volterra equations
            <https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations>`_
            on Wikipedia.
    .. [13] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
            Equations I: Nonstiff Problems", Sec. II.
    .. [14] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.

    Examples
    --------
    TODO:
    """
    t0, tf = map(float, t_span)

    if args is not None:
        # Wrap the user's fun (and jac, if given) in lambdas to hide the
        # additional parameters.  Pass in the original fun as a keyword
        # argument to keep it in the scope of the lambda.
        try:
            _ = [*(args)]
        except TypeError as exp:
            suggestion_tuple = (
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            )
            raise TypeError(suggestion_tuple) from exp

        # def fun(t, x, x_dot, fun=fun):
        #     return fun(t, x, x_dot, *args)
        def fun(t, x, fun=fun):
            return fun(t, x, *args)

        jac = options.get("jac")
        if callable(jac):
            # options["jac"] = lambda t, x: jac(t, x, *args)
            options["jac"] = lambda t, x, x_dot: jac(t, x, x_dot, *args)

    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        d = np.diff(t_eval)
        if tf > t0 and np.any(d <= 0) or tf < t0 and np.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")

        if tf > t0:
            t_eval_i = 0
        else:
            # Make order of t_eval decreasing to use np.searchsorted.
            t_eval = t_eval[::-1]
            # This will be an upper bound for slices.
            t_eval_i = t_eval.shape[0]

    assert not vectorized, "This has to be tested seperately!"
    solver = BDF(fun, t0, y0, y_dot0, tf, vectorized=vectorized, **options)

    if t_eval is None:
        ts = [t0]
        ys = [y0]
    elif t_eval is not None and dense_output:
        ts = []
        ti = [t0]
        ys = []
    else:
        ts = []
        ys = []

    interpolants = []

    events, is_terminal, event_dir = prepare_events(events)

    if events is not None:
        if args is not None:
            # Wrap user functions in lambdas to hide the additional parameters.
            # The original event function is passed as a keyword argument to the
            # lambda to keep the original function in scope (i.e., avoid the
            # late binding closure "gotcha").
            events = [lambda t, x, event=event: event(t, x, *args) for event in events]
        g = [event(t0, y0) for event in events]
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
    else:
        t_events = None
        y_events = None

    status = None
    while status is None:
        message = solver.step()

        if solver.status == "finished":
            status = 0
        elif solver.status == "failed":
            status = -1
            break

        t_old = solver.t_old
        t = solver.t
        y = solver.y

        if dense_output:
            sol = solver.dense_output()
            interpolants.append(sol)
        else:
            sol = None

        if events is not None:
            g_new = [event(t, y) for event in events]
            active_events = find_active_events(g, g_new, event_dir)
            if active_events.size > 0:
                if sol is None:
                    sol = solver.dense_output()

                root_indices, roots, terminate = handle_events(
                    sol, events, active_events, is_terminal, t_old, t
                )

                for e, te in zip(root_indices, roots):
                    t_events[e].append(te)
                    y_events[e].append(sol(te))

                if terminate:
                    status = 1
                    t = roots[-1]
                    y = sol(t)

            g = g_new

        if t_eval is None:
            ts.append(t)
            ys.append(y)
        else:
            # The value in t_eval equal to t will be included.
            if solver.direction > 0:
                t_eval_i_new = np.searchsorted(t_eval, t, side="right")
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            else:
                t_eval_i_new = np.searchsorted(t_eval, t, side="left")
                # It has to be done with two slice operations, because
                # you can't slice to 0th element inclusive using backward
                # slicing.
                t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

            if t_eval_step.size > 0:
                if sol is None:
                    sol = solver.dense_output()
                ts.append(t_eval_step)
                ys.append(sol(t_eval_step))
                t_eval_i = t_eval_i_new

        if t_eval is not None and dense_output:
            ti.append(t)

    message = MESSAGES.get(status, message)

    if t_events is not None:
        t_events = [np.asarray(te) for te in t_events]
        y_events = [np.asarray(ye) for ye in y_events]

    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
    elif ts:
        ts = np.hstack(ts)
        ys = np.hstack(ys)

    if dense_output:
        if t_eval is None:
            sol = OdeSolution(
                ts, interpolants, alt_segment=True if method in [BDF, LSODA] else False
            )
        else:
            sol = OdeSolution(
                ti, interpolants, alt_segment=True if method in [BDF, LSODA] else False
            )
    else:
        sol = None

    return OdeResult(
        t=ts,
        y=ys,
        sol=sol,
        t_events=t_events,
        y_events=y_events,
        nfev=solver.nfev,
        njev=solver.njev,
        nlu=solver.nlu,
        status=status,
        message=message,
        success=status >= 0,
    )


if __name__ == "__main__":
    mass = 1
    length = 1
    gravity = 9.81

    def fun(t, y, y_dot):
        x, y, ux, uy, la = y
        x_dot, y_dot, ux_dot, uy_dot, _ = y_dot
        return np.array(
            [
                x_dot - ux,
                y_dot - uy,
                mass * ux_dot - 2 * x * la,
                mass * uy_dot - 2 * y * la + mass * gravity,
                x**2 + y**2 - length**2,
                # la,
                # 2 * x * ux + 2 * y * uy,
                # 2 * x * x_dot + 2 * y * y_dot,
                # 2 * (x_dot * ux + x * ux_dot) + 2 * (y_dot * uy + y * uy_dot),
            ]
        )

    def jac(t, y, y_dot, c):
        from cardillo.math import approx_fprime

        return approx_fprime(y, lambda y: fun(t, y, y_dot)) + c * approx_fprime(
            y_dot, lambda y_dot: fun(t, y, y_dot)
        )

    t_span = [0, 3]
    y0 = np.array([length, 0, 0, 0, 0], dtype=float)
    y_dot0 = np.array([0, 0, 0, -gravity, 0], dtype=float)
    # y0 = np.array([length, 0, 1, 0, 0], dtype=float)
    # y_dot0 = np.array([1, 0, 0, -gravity, 0], dtype=float)
    sol = solve_dae(
        fun, t_span, y0, y_dot0, jac=jac, first_step=1e-3, max_step=1e-0
    )  # , atol=1e-5, rtol=1e-5, max_step=1e-3)

    print(f"nfev: {sol.nfev}; njev: {sol.njev}; nlu: {sol.nlu}")

    t = sol.t
    y_vec = sol.y

    x = y_vec[0]
    y = y_vec[1]
    ux = y_vec[2]
    uy = y_vec[3]
    la = y_vec[4]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    # ax.plot(x, y, "-k")
    ax.plot(t, x, "-k", label="x(t)")
    ax.plot(t, y, "--r", label="y(t)")
    ax.grid()
    ax.legend()
    plt.show()
