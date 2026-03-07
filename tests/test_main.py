import numpy as np
from t1sim import simulate_T1_rk45  


def _p1_from_rhos(rhos: np.ndarray) -> np.ndarray:
    # excited-state population p1(t) = rho_11(t)
    return np.real(rhos[:, 1, 1])


def test_p1_at_T1_is_exp_minus_1():
    rho0 = np.array([[0, 0],
                     [0, 1]], dtype=complex)  # |1><1|
    T1 = 30e-6
    t_eval = np.linspace(0, 2 * T1, 400)

    # max_step avoids issues on some SciPy versions if max_step=None is problematic
    t, rhos = simulate_T1_rk45(rho0, T1, t_eval, max_step=T1 / 200)

    p1 = _p1_from_rhos(rhos)
    idx = np.argmin(np.abs(t - T1))
    assert abs(p1[idx] - np.exp(-1)) < 2e-3


def test_trace_and_hermiticity_preserved():
    # Start from a valid mixed state (still trace-1, Hermitian)
    rho0 = np.array([[0.3, 0.2 + 0.1j],
                     [0.2 - 0.1j, 0.7]], dtype=complex)
    T1 = 40e-6
    t_eval = np.linspace(0, 3 * T1, 300)

    t, rhos = simulate_T1_rk45(rho0, T1, t_eval, max_step=T1 / 200)

    # Trace ~ 1
    traces = np.array([np.trace(r) for r in rhos])
    assert np.max(np.abs(traces - 1.0)) < 1e-8

    # Hermiticity: rho == rho^†
    herm_err = np.array([np.max(np.abs(r - r.conj().T)) for r in rhos])
    assert np.max(herm_err) < 1e-8


def test_density_matrix_is_positive_semidefinite():
    rho0 = np.array([[0.0, 0.0],
                     [0.0, 1.0]], dtype=complex)  # |1><1|
    T1 = 25e-6
    t_eval = np.linspace(0, 4 * T1, 250)

    t, rhos = simulate_T1_rk45(rho0, T1, t_eval, max_step=T1 / 200)

    # eigenvalues should be >= 0 (allow tiny numerical negatives)
    mins = []
    for r in rhos:
        evals = np.linalg.eigvalsh((r + r.conj().T) / 2)  # ensure Hermitian for eigvalsh
        mins.append(np.min(evals))
    assert min(mins) > -1e-10


def test_p1_is_monotone_decreasing_for_excited_initial_state():
    rho0 = np.array([[0, 0],
                     [0, 1]], dtype=complex)  # |1><1|
    T1 = 30e-6
    t_eval = np.linspace(0, 5 * T1, 500)

    t, rhos = simulate_T1_rk45(rho0, T1, t_eval, max_step=T1 / 300)
    p1 = _p1_from_rhos(rhos)

    # allow tiny positive bumps from numerical error
    assert np.max(np.diff(p1)) < 1e-9
