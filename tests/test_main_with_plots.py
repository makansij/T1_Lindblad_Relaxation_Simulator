import os
import sys
import subprocess
import numpy as np
import main_with_plots as mwp


def test_p1_at_T1_is_exp_minus_1():
    rho0 = np.array([[0, 0],
                     [0, 1]], dtype=complex)  # |1><1|
    T1 = 30e-6
    t_eval = np.linspace(0, 2 * T1, 400)

    t, rhos = mwp.simulate_T1_rk45(rho0, T1, t_eval, max_step=T1 / 200)
    p1 = np.real(rhos[:, 1, 1])

    idx = np.argmin(np.abs(t - T1))
    assert abs(p1[idx] - np.exp(-1)) < 2e-3


def test_trace_is_one():
    rho0 = np.array([[0.3, 0.2 + 0.1j],
                     [0.2 - 0.1j, 0.7]], dtype=complex)
    T1 = 40e-6
    t_eval = np.linspace(0, 3 * T1, 300)

    t, rhos = mwp.simulate_T1_rk45(rho0, T1, t_eval, max_step=T1 / 200)
    traces = np.array([np.trace(r) for r in rhos])
    assert np.max(np.abs(traces - 1.0)) < 1e-8


def test_script_creates_plot_files(tmp_path):
    # Run the script in a temp directory so the PNGs land there.
    script_path = os.path.abspath(mwp.__file__)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # prevents GUI windows; plt.show() becomes non-interactive

    subprocess.run(
        [sys.executable, script_path],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=60,
    )

    f1 = tmp_path / "t1_relaxation.png"
    f2 = tmp_path / "t1_populations_coherence.png"

    assert f1.exists() and f1.stat().st_size > 0
    assert f2.exists() and f2.stat().st_size > 0
