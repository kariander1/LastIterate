# EG simulation with global results dir and display flag.
# - Set RESULTS_DIR and SHOW_PLOTS at the top.
# - If SHOW_PLOTS is False, figures are saved (PNG) and *not* shown.
# - CSVs for threshold tables are also saved.
#
# Plots comply with the rules: matplotlib only, single-plot figures, no explicit colors.

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, Tuple

# -----------------------------
# Global controls
# -----------------------------
RESULTS_DIR = "./results"
SHOW_PLOTS = True  # <- set to True to display; False saves only

os.makedirs(RESULTS_DIR, exist_ok=True)

def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()

def _save_or_show(fig, filename: str):
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return out_path

# -----------------------------
# Core utilities
# -----------------------------
def project_box(z: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
    x, y = z
    x = np.minimum(np.maximum(x, xmin), xmax)
    y = np.minimum(np.maximum(y, ymin), ymax)
    return np.array([x, y], dtype=float)

def extragradient(
    F: Callable[[np.ndarray], np.ndarray],
    proj: Callable[[np.ndarray], np.ndarray],
    z0: np.ndarray,
    eta: float,
    T: int
) -> np.ndarray:
    z = z0.astype(float).copy()
    traj = np.zeros((T + 1, z0.size), dtype=float)
    traj[0] = z
    for t in range(T):
        g = F(z)
        z_half = proj(z - eta * g)
        g_half = F(z_half)
        z = proj(z - eta * g_half)
        traj[t + 1] = z
    return traj

def residual(F: Callable[[np.ndarray], np.ndarray], z: np.ndarray) -> float:
    return float(np.linalg.norm(F(z)))

def running_averages(traj: np.ndarray) -> np.ndarray:
    csum = np.cumsum(traj, axis=0)
    denominators = np.arange(1, traj.shape[0] + 1, dtype=float).reshape(-1, 1)
    return csum / denominators

def times_to_threshold(vals: np.ndarray, thresholds: list) -> dict:
    out = {}
    for thr in thresholds:
        idx = np.where(vals <= thr)[0]
        out[thr] = int(idx[0]) if idx.size > 0 else None
    return out

# -----------------------------
# Example problems
# -----------------------------
def make_problem_generic(R: float = 2.0) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Generic smooth convex–concave example:
        f(x,y) = 0.25 x^4 - 0.25 y^4 + x y
        F(z) = (x^3 + y, y^3 - x)
    Project onto [-R,R] × [-R,R].
    """
    def F(z: np.ndarray) -> np.ndarray:
        x, y = z
        return np.array([x**3 + y, y**3 - x], dtype=float)
    proj = lambda z: project_box(z, -R, R, -R, R)
    return F, proj

def make_problem_bilinear(R: float = 2.0) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Bilinear example:
        f(x,y) = x y
        F(z) = (y, -x)
    Project onto [-R,R] × [-R,R].
    """
    def F(z: np.ndarray) -> np.ndarray:
        x, y = z
        return np.array([y, -x], dtype=float)
    proj = lambda z: project_box(z, -R, R, -R, R)
    return F, proj

# -----------------------------
# Plotting helpers
# -----------------------------
def run_and_plot(F, proj, z0, eta, T, title_prefix):
    slug = _slug(title_prefix)
    traj = extragradient(F, proj, z0=z0, eta=eta, T=T)
    res_last = np.array([residual(F, traj[t]) for t in range(traj.shape[0])])
    traj_avg = running_averages(traj)
    res_avg = np.array([residual(F, traj_avg[t]) for t in range(traj_avg.shape[0])])

    # Plot 1: last-iterate residual
    fig = plt.figure()
    plt.loglog(np.arange(traj.shape[0]), res_last + 1e-16)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Residual ||F(z^t)||")
    plt.title(f"{title_prefix}: Last-iterate residual")
    last_png = _save_or_show(fig, f"{slug}_last_iter_residual.png")

    # Plot 2: averaged-iterate residual
    fig = plt.figure()
    plt.loglog(np.arange(traj_avg.shape[0]), res_avg + 1e-16)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Residual ||F(\\bar z^t)||")
    plt.title(f"{title_prefix}: Averaged-iterate residual")
    avg_png = _save_or_show(fig, f"{slug}_avg_iter_residual.png")

    # Plot 3: 2D trajectory
    fig = plt.figure()
    plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=2, linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title_prefix}: Trajectory in parameter space")
    traj_png = _save_or_show(fig, f"{slug}_trajectory_2d.png")

    # Threshold report
    thresholds = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    times_last = times_to_threshold(res_last, thresholds)
    times_avg = times_to_threshold(res_avg, thresholds)

    df = pd.DataFrame(
        {
            "threshold": thresholds,
            "first_iter_last_iterate": [times_last[t] for t in thresholds],
            "first_iter_averaged_iterate": [times_avg[t] for t in thresholds],
        }
    )
    csv_path = os.path.join(RESULTS_DIR, f"{slug}_time_to_threshold.csv")
    df.to_csv(csv_path, index=False)

    # Optionally display table to the user
    if SHOW_PLOTS:
        try:
            from caas_jupyter_tools import display_dataframe_to_user
            display_dataframe_to_user(f"{title_prefix} — time-to-threshold", df)
        except Exception:
            pass

    return {
        "traj": traj,
        "res_last": res_last,
        "res_avg": res_avg,
        "paths": {
            "last_residual_png": last_png,
            "avg_residual_png": avg_png,
            "traj2d_png": traj_png,
            "thresholds_csv": csv_path,
        },
    }



def plot_f_surface_with_trajectory(f_scalar, traj, R, title_prefix, grid_n=120):
    xs = np.linspace(-R, R, grid_n)
    ys = np.linspace(-R, R, grid_n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    Z = f_scalar(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.85)

    Z_traj = f_scalar(traj[:, 0], traj[:, 1])
    ax.plot(traj[:, 0], traj[:, 1], Z_traj, linewidth=2)
    ax.scatter(traj[:, 0], traj[:, 1], Z_traj, s=6)
    ax.scatter(traj[0, 0], traj[0, 1], Z_traj[0], s=60)
    ax.scatter(traj[-1, 0], traj[-1, 1], Z_traj[-1], s=60)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
    ax.set_title(f"{title_prefix}: 3D surface of f(x,y) with EG iterates")
    return _save_or_show(fig, f"{_slug(title_prefix)}_surface_f3d.png")


def plot_F_surface_with_trajectory(F, traj, R, title_prefix, grid_n=80):
    slug = _slug(title_prefix)
    xs = np.linspace(-R, R, grid_n)
    ys = np.linspace(-R, R, grid_n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    Z = np.empty_like(X)
    for i in range(grid_n):
        for j in range(grid_n):
            z = np.array([X[i, j], Y[i, j]])
            val = F(z)
            Z[i, j] = np.sqrt(val[0]**2 + val[1]**2)

    Z_traj = np.array([np.linalg.norm(F(traj[t])) for t in range(traj.shape[0])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.85)
    ax.plot(traj[:, 0], traj[:, 1], Z_traj, linewidth=2)
    ax.scatter(traj[:, 0], traj[:, 1], Z_traj, s=6)
    ax.scatter(traj[0, 0], traj[0, 1], Z_traj[0], s=60)
    ax.scatter(traj[-1, 0], traj[-1, 1], Z_traj[-1], s=60)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$\|F(x,y)\|$")
    ax.set_title(f"{title_prefix}: 3D surface of $\\|F(x,y)\\|$ with EG iterates")
    surf_png = _save_or_show(fig, f"{slug}_surface_traj3d.png")
    return {"surface_traj3d_png": surf_png}

def plot_F_quiver_with_trajectory(F, traj, R, title_prefix, grid_n=20):
    xs = np.linspace(-R, R, grid_n)
    ys = np.linspace(-R, R, grid_n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for i in range(grid_n):
        for j in range(grid_n):
            g = F(np.array([X[i, j], Y[i, j]]))
            U[i, j], V[i, j] = g[0], g[1]

    fig = plt.figure()
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy')
    plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=2, linewidth=1)
    plt.scatter([traj[0, 0]], [traj[0, 1]], s=60)
    plt.scatter([traj[-1, 0]], [traj[-1, 1]], s=60)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"{title_prefix}: Vector field F(x,y) with EG trajectory")
    return _save_or_show(fig, f"{_slug(title_prefix)}_F_quiver_2d.png")

def plot_F_space_path(F, traj, title_prefix):
    Fx = np.array([F(z)[0] for z in traj])
    Fy = np.array([F(z)[1] for z in traj])
    fig = plt.figure()
    plt.plot(Fx, Fy, marker='o', markersize=2, linewidth=1)
    plt.scatter([Fx[0]], [Fy[0]], s=60)
    plt.scatter([Fx[-1]], [Fy[-1]], s=60)
    plt.xlabel(r"$F_x(z)$"); plt.ylabel(r"$F_y(z)$")
    plt.title(f"{title_prefix}: Path in F-space")
    return _save_or_show(fig, f"{_slug(title_prefix)}_F_space_path.png")

def f_generic(x, y):
    # matches F(x,y) = (x**3 + y, y**3 - x)
    return 0.25 * x**4 - 0.25 * y**4 + x * y

def f_bilinear(x, y):
    # matches F(x,y) = (y, -x)
    return x * y

# -----------------------------
# Run experiments
# -----------------------------
T = 2000
z0 = np.array([1.5, -1.0], dtype=float)

# 1) Generic smooth convex–concave
F_gen, proj_gen = make_problem_generic(R=2.0)
results_generic = run_and_plot(F_gen, proj_gen, z0=z0, eta=0.05, T=T, title_prefix="Generic convex–concave")
generic_3d = plot_F_surface_with_trajectory(F_gen, results_generic["traj"], R=2.0, title_prefix="Generic convex–concave")

# 2) Bilinear sanity check
F_bil, proj_bil = make_problem_bilinear(R=2.0)
results_bilinear = run_and_plot(F_bil, proj_bil, z0=z0, eta=0.2, T=T, title_prefix="Bilinear f(x,y)=x y")
bilinear_3d = plot_F_surface_with_trajectory(F_bil, results_bilinear["traj"], R=2.0, title_prefix="Bilinear f(x,y)=x y")

R_box = 2.0  # same box you used for projection
# Generic
plot_f_surface_with_trajectory(f_generic, results_generic["traj"], R=R_box, title_prefix="Generic convex–concave")
plot_F_quiver_with_trajectory(F_gen, results_generic["traj"], R=R_box, title_prefix="Generic convex–concave")
plot_F_space_path(F_gen, results_generic["traj"], title_prefix="Generic convex–concave")

# Bilinear
plot_f_surface_with_trajectory(f_bilinear, results_bilinear["traj"], R=R_box, title_prefix="Bilinear f(x,y)=x y")
plot_F_quiver_with_trajectory(F_bil, results_bilinear["traj"], R=R_box, title_prefix="Bilinear f(x,y)=x y")
plot_F_space_path(F_bil, results_bilinear["traj"], title_prefix="Bilinear f(x,y)=x y")


# Summarize saved outputs
saved_files = [
    results_generic["paths"]["last_residual_png"],
    results_generic["paths"]["avg_residual_png"],
    results_generic["paths"]["traj2d_png"],
    results_generic["paths"]["thresholds_csv"],
    generic_3d["surface_traj3d_png"],
    results_bilinear["paths"]["last_residual_png"],
    results_bilinear["paths"]["avg_residual_png"],
    results_bilinear["paths"]["traj2d_png"],
    results_bilinear["paths"]["thresholds_csv"],
    bilinear_3d["surface_traj3d_png"],
]
saved_files
