import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from t1sim import simulate_T1_rk45

# Pauli operators (computational basis |0>, |1>)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Z = np.array([[1, 0],
              [0, -1]], dtype=complex)


def exp_x(rho: np.ndarray) -> float:
    """Return <X> = Tr(rho X) as a real number."""
    return float(np.real(np.trace(rho @ X)))


if __name__ == "__main__":
    # --- Parameters ---
    T1 = 30e-6  # seconds (hardware-measured relaxation time)
    Delta_hz = 0.5e6  # detuning in Hz for Ramsey-style Z precession
    Delta = 2 * np.pi * Delta_hz  # convert to rad/s

    # Effective Hamiltonian in rotating frame: H = (Δ/2) Z
    H = 0.5 * Delta * Z

    # Time grid
    t_eval = np.linspace(0, 60e-6, 2000)  # seconds

    # Initial state: |+><+| (superposition)
    rho0 = np.array([[0.5, 0.5],
                     [0.5, 0.5]], dtype=complex)

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
        max_step=(t_eval[-1] - t_eval[0]) / 5000,  # keep steps reasonably small
    )

    # --- Observables ---
    x_sim = np.array([exp_x(r) for r in rhos])

    # Expected Ramsey signal with T1-only (no Tphi/T2):
    # <X>(t) = exp(-t/(2T1)) * cos(Δ t)
    x_expected = np.exp(-t / (2 * T1)) * np.cos(Delta * t)

    # Populations (also have simple expectations for T1-only here)
    p1_sim = np.real(rhos[:, 1, 1])
    p0_sim = np.real(rhos[:, 0, 0])

    # Starting from p1(0)=0.5, T1 decay gives p1(t)=0.5 exp(-t/T1)
    p1_expected = 0.5 * np.exp(-t / T1)
    p0_expected = 1.0 - p1_expected

    t_us = t * 1e6

    # --- Figure 1: Ramsey <X> signal ---
    plt.figure(figsize=(7, 4))
    plt.plot(t_us, x_sim, label="Simulated ⟨X⟩(t)")
    plt.plot(t_us, x_expected, "--", label="Expected exp(-t/(2T1))·cos(Δt)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Value")
    plt.title("Ramsey with T1 Relaxation (Lindblad, rotating frame)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "ramsey_x.png", dpi=200)

    # --- Figure 2: Populations ---
    plt.figure(figsize=(7, 4))
    plt.plot(t_us, p0_sim, label="Simulated ρ00(t)")
    plt.plot(t_us, p0_expected, "--", label="Expected 1-0.5·exp(-t/T1)")
    plt.plot(t_us, p1_sim, label="Simulated ρ11(t)")
    plt.plot(t_us, p1_expected, "--", label="Expected 0.5·exp(-t/T1)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Probability")
    plt.title("Populations under T1 Relaxation (Lindblad, H=ΔZ/2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "ramsey_populations.png", dpi=200)

    plt.show()
