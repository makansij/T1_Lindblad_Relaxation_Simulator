import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from t1sim import simulate_T1_rk45

# Pauli operators (computational basis |0>, |1>)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Z = np.array([[1, 0],
              [0, -1]], dtype=complex)


def exp_z(rho: np.ndarray) -> float:
    """Return <Z> = Tr(rho Z) as a real number."""
    return float(np.real(np.trace(rho @ Z)))


if __name__ == "__main__":
    # --- Parameters ---
    T1 = 30e-6  # seconds (hardware-measured relaxation time)
    Omega_hz = 1.0e6  # Rabi frequency (Hz) for resonant X-drive
    Omega = 2 * np.pi * Omega_hz  # convert to rad/s

    # Effective Hamiltonian in rotating frame (resonant drive): H = (Ω/2) X
    H = 0.5 * Omega * X

    # Time grid: a few microseconds covers multiple oscillations at ~1 MHz
    t_eval = np.linspace(0, 10e-6, 2000)  # seconds

    # Initial state: |0><0|
    rho0 = np.array([[1.0, 0.0],
                     [0.0, 0.0]], dtype=complex)

    # Artifacts directory under repo root (works when script is in examples/)
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # --- Simulate: dρ/dt = -i[H, ρ] + D[ sqrt(1/T1) σ- ](ρ) ---
    t, rhos = simulate_T1_rk45(
        rho0=rho0,
        T1=T1,
        t_eval=t_eval,
        H=H,
        max_step=(t_eval[-1] - t_eval[0]) / 5000,
    )

    # --- Observables ---
    p1_sim = np.real(rhos[:, 1, 1])
    p0_sim = np.real(rhos[:, 0, 0])
    z_sim = np.array([exp_z(r) for r in rhos])

    # Rough expected Rabi behavior with T1-only:
    # p1(t) ≈ (1 - cos(Ω t))/2 * exp(-t/(2T1))   (heuristic envelope)
    # This is not an exact closed-form for the driven-damped master equation,
    # but is useful as a qualitative reference.
    p1_expected = 0.5 * (1 - np.cos(Omega * t)) * np.exp(-t / (2 * T1))

    t_us = t * 1e6

    # --- Figure 1: Rabi oscillations in excited-state population ---
    plt.figure(figsize=(7, 4))
    plt.plot(t_us, p1_sim, label="Simulated ρ11(t)")
    plt.plot(t_us, p1_expected, "--", label="Heuristic envelope: 0.5(1-cos(Ωt))exp(-t/(2T1))")
    plt.xlabel("Time (µs)")
    plt.ylabel("Probability")
    plt.title("Rabi with T1 Relaxation (Lindblad, rotating frame)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "rabi_p1.png", dpi=200)

    # --- Figure 2: Populations and <Z> ---
    plt.figure(figsize=(7, 4))
    plt.plot(t_us, p0_sim, label="Simulated ρ00(t)")
    plt.plot(t_us, p1_sim, label="Simulated ρ11(t)")
    plt.plot(t_us, z_sim, label="Simulated ⟨Z⟩(t)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Value")
    plt.title("Populations and ⟨Z⟩ under Rabi drive + T1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "rabi_populations_z.png", dpi=200)

    plt.show()
