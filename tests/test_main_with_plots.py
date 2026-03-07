import os, sys, subprocess
from pathlib import Path
import numpy as np
from t1sim import simulate_T1_rk45


def test_p1_at_T1_is_exp_minus_1():
    rho0 = np.array([[0, 0],
                     [0, 1]], dtype=complex)  # |1><1|
    T1 = 30e-6
    t_eval = np.linspace(0, 2 * T1, 400)

    t, rhos = simulate_T1_rk45(rho0, T1, t_eval, max_step=T1 / 200)
    p1 = np.real(rhos[:, 1, 1])

    idx = np.argmin(np.abs(t - T1))
    assert abs(p1[idx] - np.exp(-1)) < 2e-3


def test_trace_is_one():
    rho0 = np.array([[0.3, 0.2 + 0.1j],
                     [0.2 - 0.1j, 0.7]], dtype=complex)
    T1 = 40e-6
    t_eval = np.linspace(0, 3 * T1, 300)

    t, rhos = simulate_T1_rk45(rho0, T1, t_eval, max_step=T1 / 200)
    traces = np.array([np.trace(r) for r in rhos])
    assert np.max(np.abs(traces - 1.0)) < 1e-8


def test_script_creates_plot_files(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]          # repo root
    script_path = repo_root / "examples" / "t1_with_plots.py"

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # headless plotting

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,          # so PNGs are created in tmp_path
        env=env,
        check=True,
        timeout=60,
    )

    assert (tmp_path / "t1_relaxation.png").exists()
    assert (tmp_path / "t1_populations_coherence.png").exists()
