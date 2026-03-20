# src/t1sim/sim.py
import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Union

sigma_minus = np.array([[0, 1],
                        [0, 0]], dtype=complex)  # |0><1|

def lindblad_dissipator(rho: np.ndarray, L: np.ndarray) -> np.ndarray:
    """D[L](rho) = L rho L† - 1/2 (L†L rho + rho L†L)"""
    Ld = L.conj().T
    LdL = Ld @ L
    return L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)

def pack_complex_matrix_to_real_vec(A: np.ndarray) -> np.ndarray:
    v = A.reshape(-1)
    return np.concatenate([v.real, v.imag])

def unpack_real_vec_to_complex_matrix(y: np.ndarray, dim: int) -> np.ndarray:
    n = dim * dim
    v = y[:n] + 1j * y[n:]
    return v.reshape((dim, dim))

def simulate_T1_rk45(
    rho0: np.ndarray,
    T1: float,
    t_eval: np.ndarray,
    H: Union[np.ndarray, Callable[[float], np.ndarray], None] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float | None = None,
):
    """
    Lindblad evolution with T1 relaxation and an optional Hamiltonian term:
        dρ/dt = -i[H, ρ] + D[L](ρ),  L = sqrt(1/T1) * σ-
    If H is None, it defaults to 0 (no coherent evolution).
    """
    if rho0.shape != (2, 2):
        raise ValueError("Expected single-qubit rho0 of shape (2,2).")
    if T1 <= 0:
        raise ValueError("T1 must be positive.")
    if t_eval.ndim != 1 or len(t_eval) < 2:
        raise ValueError("t_eval must be a 1D array with at least 2 time points.")

    gamma1 = 1.0 / T1
    L = np.sqrt(gamma1) * sigma_minus
    dim = 2

    # Hamiltonian: allow a constant 2x2 matrix or a callable H(t). Default is H=0.
    if H is None:
        H_const = np.zeros((2, 2), dtype=complex)
        H_of_t = None
    elif callable(H):
        H_const = None
        H_of_t = H
    else:
        H_const = np.asarray(H, dtype=complex)
        if H_const.shape != (2, 2):
            raise ValueError("H must have shape (2,2) for a single-qubit simulation.")
        H_of_t = None

    y0 = pack_complex_matrix_to_real_vec(rho0)

    def rhs(t, y):
        rho = unpack_real_vec_to_complex_matrix(y, dim)

        # Evaluate Hamiltonian (constant or time-dependent)
        if H_of_t is None:
            Ht = H_const
        else:
            Ht = np.asarray(H_of_t(float(t)), dtype=complex)
            if Ht.shape != (2, 2):
                raise ValueError("H(t) must return a (2,2) matrix for a single-qubit simulation.")

        unitary = -1j * (Ht @ rho - rho @ Ht)
        drho = unitary + lindblad_dissipator(rho, L)
        return pack_complex_matrix_to_real_vec(drho)

    kwargs = dict(method="RK45", rtol=rtol, atol=atol, t_eval=t_eval)
    if max_step is not None:
        kwargs["max_step"] = max_step  # avoids SciPy versions that choke on None

    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), y0, **kwargs)
    if not sol.success:
        raise RuntimeError(sol.message)

    rhos = np.array([unpack_real_vec_to_complex_matrix(y, dim) for y in sol.y.T])
    return sol.t, rhos
