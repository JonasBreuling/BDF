import numpy as np
from scipy.optimize._numdiff import approx_derivative
from BDF import solve_dae

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
        # from cardillo.math import approx_fprime
        # return approx_fprime(y, lambda y: fun(t, y, y_dot)) + c * approx_fprime(
        #     y_dot, lambda y_dot: fun(t, y, y_dot)
        # )
        return approx_derivative(lambda y: fun(t, y, y_dot), y) + c * approx_derivative(
            lambda y_dot: fun(t, y, y_dot), y_dot
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
