import numpy as np
from t1sim import simulate_T1_rk45

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
