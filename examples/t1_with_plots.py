import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Operators (computational basis |0>, |1>) ---
sigma_minus = np.array([[0, 1],
                        [0, 0]], dtype=complex)  # |0><1|

def lindblad_dissipator(rho: np.ndarray, L: np.ndarray) -> np.ndarray:
    """D[L](rho) = L rho L† - 1/2 (L†L rho + rho L†L)"""
    Ld = L.conj().T
    LdL = Ld @ L
    return L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)

def pack_complex_matrix_to_real_vec(A: np.ndarray) -> np.ndarray:
    """Flatten complex matrix -> real vector [Re; Im]."""
    v = A.reshape(-1)
    return np.concatenate([v.real, v.imag])

def unpack_real_vec_to_complex_matrix(y: np.ndarray, dim: int) -> np.ndarray:
    """Real vector [Re; Im] -> complex matrix."""
    n = dim * dim
    v = y[:n] + 1j * y[n:]
    return v.reshape((dim, dim))

def simulate_T1_rk45(rho0: np.ndarray, T1: float, t_eval: np.ndarray,
                     rtol: float = 1e-8, atol: float = 1e-10,
                     max_step: float | None = None):
    """
    Simulate T1-only relaxation with H=0:
        dρ/dt = D[L](ρ),  L = sqrt(1/T1) * sigma_minus

    Inputs:
      rho0   : (2,2) complex density matrix at t=0
      T1     : relaxation time (seconds)
      t_eval : 1D array of times (seconds) where you want ρ(t)

    Returns:
      t      : times (same as t_eval)
      rhos   : array of shape (len(t_eval), 2, 2) with ρ(t)
    """
    if rho0.shape != (2, 2):
        raise ValueError("This minimal example expects a single-qubit rho0 of shape (2,2).")
    if T1 <= 0:
        raise ValueError("T1 must be positive.")

    gamma1 = 1.0 / T1
    L = np.sqrt(gamma1) * sigma_minus
    dim = 2

    y0 = pack_complex_matrix_to_real_vec(rho0)

    def rhs(t, y):
        rho = unpack_real_vec_to_complex_matrix(y, dim)
        drho = lindblad_dissipator(rho, L)
        return pack_complex_matrix_to_real_vec(drho)
    kwargs = dict(method="RK45", rtol=rtol, atol=atol, t_eval=t_eval)
    if max_step is not None:
        kwargs["max_step"] = max_step
    
    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), y0, **kwargs)
#    sol = solve_ivp(
#        rhs,
#        t_span=(float(t_eval[0]), float(t_eval[-1])),
#        y0=y0,
#        t_eval=t_eval,
#        method="RK45",
#        rtol=rtol,
#        atol=atol,
#        max_step=max_step
#    )
    if not sol.success:
        raise RuntimeError(sol.message)

    rhos = np.array([unpack_real_vec_to_complex_matrix(y, dim) for y in sol.y.T])
    return sol.t, rhos

# -------------------- Example usage --------------------
if __name__ == "__main__":
    # Start in |1><1|
    rho0 = np.array([[0, 0],
                     [0, 1]], dtype=complex)

    T1 = 30e-6
    t_eval = np.linspace(0, 100e-6, 400)

    t, rhos = simulate_T1_rk45(rho0, T1, t_eval)

    # Excited-state population p1(t) = rho_11(t)
    p1 = np.real(rhos[:, 1, 1])

    # Quick print of a couple values
    print("p1(0) =", p1[0])
    print("p1(T1) ~", p1[np.argmin(np.abs(t - T1))], "(expected ~ e^-1 =", np.exp(-1), ")")

    # --- Plots ---
    # Compare numerical p1(t) vs the analytic expectation exp(-t/T1)
    p1_expected = np.exp(-t / T1)

    plt.figure(figsize=(7, 4))
    plt.plot(t * 1e6, p1, label='Simulated p1(t)')
    plt.plot(t * 1e6, p1_expected, '--', label='Expected exp(-t/T1)')
    plt.xlabel('Time (µs)')
    plt.ylabel('Excited-state probability p1')
    plt.title('T1 Relaxation (Lindblad, H=0)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('t1_relaxation.png', dpi=200)

    # Optional: show populations and coherence magnitude
    rho00 = np.real(rhos[:, 0, 0])
    coh01 = np.abs(rhos[:, 0, 1])
    coh_expected = coh01[0] * np.exp(-t / (2 * T1))  # expected for T1-only

    plt.figure(figsize=(7, 4))
    plt.plot(t * 1e6, rho00, label='rho00(t)')
    plt.plot(t * 1e6, p1, label='rho11(t)')
    plt.plot(t * 1e6, coh01, label='|rho01(t)|')
    plt.plot(t * 1e6, coh_expected, '--', label='|rho01(0)| exp(-t/(2T1))')
    plt.xlabel('Time (µs)')
    plt.ylabel('Value')
    plt.title('Populations and Coherence (T1-only)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('t1_populations_coherence.png', dpi=200)

    plt.show()