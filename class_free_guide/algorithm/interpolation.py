import torch
from torch import Tensor


def _thomas_solve(
    main: Tensor,
    lower: Tensor,
    upper: Tensor,
    rhs: Tensor,
) -> Tensor:
    """
    Thomas algorithm for tridiagonal system with batched RHS.

    Args:
        main:  [K]        main diagonal
        lower: [K-1]      sub-diagonal
        upper: [K-1]      super-diagonal
        rhs:   [*, K, D]  right-hand sides
    Returns:
        [*, K, D] solution
    """
    K = main.shape[0]
    if K == 1:
        return rhs / main[0]

    c = torch.zeros(K, device=main.device, dtype=main.dtype)
    d = rhs.clone()

    # forward elimination
    c[0] = upper[0] / main[0]
    d[..., 0, :] /= main[0]

    for i in range(1, K):
        denom = main[i] - lower[i - 1] * c[i - 1]
        if i < K - 1:
            c[i] = upper[i] / denom
        d[..., i, :] = (d[..., i, :] - lower[i - 1] * d[..., i - 1, :]) / denom

    # back substitution
    for i in range(K - 2, -1, -1):
        d[..., i, :] -= c[i] * d[..., i + 1, :]

    return d


def cubic_spline_interpolation(
    x_knots: Tensor,
    y_knots: Tensor,
    x_query: Tensor,
    return_derivative: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Natural cubic spline interpolation (batched, PyTorch, differentiable).

    Natural boundary: S''(x_0) = S''(x_N) = 0

    Args:
        x_knots: [N]      sorted knot x-coordinates (strictly increasing)
        y_knots: [*, N, D] values at knot points (* = arbitrary batch dims)
        x_query: [M]      query x-coordinates
        return_derivative: if True, also return first derivative dy/dx

    Returns:
        y_query:  [*, M, D] interpolated values
        dy_query: [*, M, D] first derivatives (only when return_derivative=True)

    Example::

        t = torch.linspace(0, 1, 5)                # [5]
        traj = torch.randn(32, 5, 7)               # [B, 5, D]
        t_q = torch.linspace(0, 1, 50)             # [50]
        traj_q = cubic_spline_interpolation(t, traj, t_q)  # [32, 50, 7]
    """
    N = x_knots.shape[0]
    assert N >= 2, "Need at least 2 knot points"

    h = x_knots[1:] - x_knots[:-1]  # [N-1]
    h_col = h.unsqueeze(-1)  # [N-1, 1]

    # finite-difference slopes
    dy = y_knots[..., 1:, :] - y_knots[..., :-1, :]  # [*, N-1, D]
    slopes = dy / h_col  # [*, N-1, D]

    # solve for second derivatives m at each knot (natural boundary → m_0 = m_{N-1} = 0)
    if N <= 2:
        m = torch.zeros_like(y_knots)  # [*, N, D]
    else:
        rhs = 6.0 * (slopes[..., 1:, :] - slopes[..., :-1, :])  # [*, N-2, D]
        main_diag = 2.0 * (h[:-1] + h[1:])  # [N-2]
        off_diag = h[1:-1]  # [N-3]
        m_interior = _thomas_solve(main_diag, off_diag, off_diag, rhs)  # [*, N-2, D]
        zeros = torch.zeros(*rhs.shape[:-2], 1, rhs.shape[-1], device=rhs.device, dtype=rhs.dtype)
        m = torch.cat([zeros, m_interior, zeros], dim=-2)  # [*, N, D]

    # spline coefficients per interval  S_i(x) = a + b*dx + c*dx^2 + d*dx^3
    m_i = m[..., :-1, :]  # [*, N-1, D]
    m_ip1 = m[..., 1:, :]  # [*, N-1, D]
    a = y_knots[..., :-1, :]  # [*, N-1, D]
    b = slopes - h_col * (2.0 * m_i + m_ip1) / 6.0
    c = m_i / 2.0
    d = (m_ip1 - m_i) / (6.0 * h_col)

    # locate intervals via binary search
    idx = torch.searchsorted(x_knots, x_query, right=True) - 1  # [M]
    idx = idx.clamp(0, N - 2)

    dx = (x_query - x_knots[idx]).unsqueeze(-1)  # [M, 1]

    # gather coefficients for each query point
    a_q = a[..., idx, :]  # [*, M, D]
    b_q = b[..., idx, :]
    c_q = c[..., idx, :]
    d_q = d[..., idx, :]

    # Horner evaluation
    y_query = a_q + dx * (b_q + dx * (c_q + dx * d_q))

    if return_derivative:
        dy_query = b_q + dx * (2.0 * c_q + 3.0 * dx * d_q)
        return y_query, dy_query
    return y_query
