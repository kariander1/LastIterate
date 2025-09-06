import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, Tuple
plt.rcParams["text.usetex"] = False   # use mathtext, no external LaTeX needed
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
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

def make_problem_block_bilinear(R: float = 2.0, lambdas: list = [1.0, 2.0]) -> Tuple[Callable, Callable]:
    """
    Higher-dimensional bilinear example with block-diagonal skew-symmetric operator:
        Each block corresponds to f_j(x,y) = λ_j x y.

    Args:
        R (float): Projection bound for box constraint.
        lambdas (list of float): List of λ_j values, one per 2D block.

    Returns:
        F (Callable): Monotone operator F(z) in R^{2m}.
        proj (Callable): Projection onto [-R, R]^{2m}.
    """
    dim = 2 * len(lambdas)

    def F(z: np.ndarray) -> np.ndarray:
        z = z.reshape(-1, 2)  # group into (x,y) blocks
        out = []
        for (x, y), lam in zip(z, lambdas):
            out.append([lam * y, -lam * x])
        return np.array(out, dtype=float).reshape(-1)

    def proj(z: np.ndarray) -> np.ndarray:
        return np.clip(z, -R, R)

    return F, proj

def make_problem_bilinear(R: float = 2.0, lam: float = 1.0) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Bilinear example with parameter λ:
        f(x,y) = λ x y
        F(z) = (λ y, -λ x)
    Project onto [-R,R] × [-R,R].
    
    Args:
        R (float): Projection bound for box constraint.
        lam (float): Scaling parameter λ for the bilinear function.
    
    Returns:
        F (Callable): Monotone operator F(z).
        proj (Callable): Projection onto the box [-R,R]^2.
    """
    def F(z: np.ndarray) -> np.ndarray:
        x, y = z
        return np.array([lam * y, -lam * x], dtype=float)

    proj = lambda z: project_box(z, -R, R, -R, R)
    return F, proj


# -----------------------------
# Plotting helpers
# -----------------------------
def run_and_plot(F, proj, z0, eta, T, R, title_prefix, f_scalar=None):
    slug = _slug(title_prefix)
    traj = extragradient(F, proj, z0=z0, eta=eta, T=T)
    res_last = np.array([residual(F, traj[t]) for t in range(traj.shape[0])])
    traj_avg = running_averages(traj)
    res_avg = np.array([residual(F, traj_avg[t]) for t in range(traj_avg.shape[0])])

    # Plot 1: Combined residuals (last and averaged)
    fig = plt.figure()
    plt.loglog(np.arange(traj.shape[0]), res_last + 1e-16, label="Last iterate", color='blue')
    plt.loglog(np.arange(traj_avg.shape[0]), res_avg + 1e-16, label="Averaged iterate", color='orange')
    plt.xlabel("Iterations (t)")
    plt.ylabel("Residual ||F(z)||")
    # plt.ylim(bottom=1e-4, top=0.15e1)
    # plt.title(f"{title_prefix}: Residuals")
    plt.legend(loc="lower left")
    combined_png = _save_or_show(fig, f"{slug}_combined_residuals.pdf")
    
    # Plot 3: Combined 2D trajectories (last iterate and averaged)
    fig = plt.figure()
    plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=2, linewidth=1,
             label="Last iterate", color='blue')
    plt.plot(traj_avg[:, 0], traj_avg[:, 1], marker='x', markersize=2, linewidth=1,
             label="Averaged iterate", color='orange')

    # Add faint dashed unit circle for reference
    theta = np.linspace(0, 2*np.pi, 500)
    radius = 1.0  # fixed unit circle
    plt.plot(radius*np.cos(theta), radius*np.sin(theta),
             linestyle='--', color='gray', alpha=0.5, linewidth=1)

    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title(f"{title_prefix}: Trajectories in parameter space")
    plt.legend(loc="upper left")
    plt.axis("equal")

    traj_combined_png = _save_or_show(fig, f"{slug}_trajectory_2d_combined.pdf")
    
        
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

    return {
        "traj": traj,
        "traj_avg": traj_avg,
        "res_last": res_last,
        "res_avg": res_avg,
        "paths": {
            "combined_residuals_png": combined_png,
            "traj2d_png": traj_combined_png,
            "thresholds_csv": csv_path,
        },
    }


def plot_f_surface_with_trajectory(f_scalar, traj, traj_avg, R, title_prefix, grid_n=120):
    # Enable LaTeX-like text rendering for all math

    xs = np.linspace(-R, R, grid_n)
    ys = np.linspace(-R, R, grid_n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    Z = f_scalar(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Transparent surface
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.5, cmap="viridis")

    # Last iterate trajectory
    Z_traj = f_scalar(traj[:, 0], traj[:, 1])
    ax.plot(traj[:, 0], traj[:, 1], Z_traj, linewidth=2, color="blue", label=r"Last iterate")

    # Averaged iterate trajectory
    Z_traj_avg = f_scalar(traj_avg[:, 0], traj_avg[:, 1])
    ax.plot(traj_avg[:, 0], traj_avg[:, 1], Z_traj_avg, linewidth=2, color="orange", label=r"Averaged iterate")

    # Projected 2D unit circle onto the surface
    theta = np.linspace(0, 2*np.pi, 500)
    radius = 1.0
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    circle_z = f_scalar(circle_x, circle_y)
    ax.plot(circle_x, circle_y, circle_z,
            linestyle='--', color='gray', alpha=0.6, linewidth=1.2)

    # Labels and title with math mode
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$f(x,y)$")
    # ax.set_title(rf"{title_prefix}: 3D surface of $f(x,y)$ with EG iterates")
    ax.legend(loc="upper left")
    ax.set_xlim([-1.5, 1.5])   # zoom into x range
    ax.set_ylim([-1.5, 1.5])   # zoom into y range
    ax.set_zlim([-1.5, 1.5])   # zoom into z range
    ax.view_init(elev=49)   # elevation=30°, azimuth=45°

    return _save_or_show(fig, f"{_slug(title_prefix)}_surface_f3d.pdf")


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
    surf_png = _save_or_show(fig, f"{slug}_surface_traj3d.pdf")
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
    return _save_or_show(fig, f"{_slug(title_prefix)}_F_quiver_2d.pdf")

def plot_F_space_path(F, traj, title_prefix):
    Fx = np.array([F(z)[0] for z in traj])
    Fy = np.array([F(z)[1] for z in traj])
    fig = plt.figure()
    plt.plot(Fx, Fy, marker='o', markersize=2, linewidth=1)
    plt.scatter([Fx[0]], [Fy[0]], s=60)
    plt.scatter([Fx[-1]], [Fy[-1]], s=60)
    plt.xlabel(r"$F_x(z)$"); plt.ylabel(r"$F_y(z)$")
    # plt.title(f"{title_prefix}: Path in F-space")
    return _save_or_show(fig, f"{_slug(title_prefix)}_F_space_path.pdf")


def f_bilinear(x, y, lam=1.0):
    # matches F(x,y) = (y, -x)
    return  lam * x * y


# -----------------------------
# Run experiments
# -----------------------------
T = 1000
z0 = np.array([np.sqrt(2)/2, np.sqrt(2)/2], dtype=float)
eigen_lambda = 1.0  # scaling for bilinear
R_box = 2.0


F_bil, proj_bil = make_problem_bilinear(R=R_box, lam=eigen_lambda)
f_bilinear_scalar = lambda x, y: f_bilinear(x, y, lam=eigen_lambda)
eta_stalled = 1/eigen_lambda
eta_slow = 1/eigen_lambda -  0.01
eta_fast = eigen_lambda/10

results_bilinear = run_and_plot(F_bil, proj_bil, z0=z0, eta=eta_stalled, T=T, R=R_box, title_prefix="Stalled Bilinear f(x,y)=x y")
plot_f_surface_with_trajectory(f_bilinear_scalar, results_bilinear["traj"], results_bilinear["traj_avg"], R=R_box, title_prefix="Stalled Bilinear f(x,y)=x y")


results_bilinear = run_and_plot(F_bil, proj_bil, z0=z0, eta=eta_slow, T=T, R=R_box, title_prefix="Slow Bilinear f(x,y)=x y")
plot_f_surface_with_trajectory(f_bilinear_scalar, results_bilinear["traj"], results_bilinear["traj_avg"], R=R_box, title_prefix="Slow Bilinear f(x,y)=x y")


results_bilinear = run_and_plot(F_bil, proj_bil, z0=z0, eta=eta_fast, T=T, R=R_box, title_prefix="Fast Bilinear f(x,y)=x y")
plot_f_surface_with_trajectory(f_bilinear_scalar, results_bilinear["traj"], results_bilinear["traj_avg"], R=R_box, title_prefix="Fast Bilinear f(x,y)=x y")


R_box = 4.0  # box for projection
# Define a 4D bilinear problem with two different frequencies
lambdas = np.geomspace(0.1, 10.0, num=6)  # 6 blocks from 0.1 to 10

F_block, proj_block = make_problem_block_bilinear(R=2.0, lambdas=lambdas)

# Initial point in R^4
z0_block = np.random.randn(2 * len(lambdas))  # random initial point

# Run extragradient
T = 10000

etas = [0.005, 0.01, 0.1, 0.5, 1.0, 1.5]

for eta in etas:
    run_and_plot(F_block, proj_block, z0=z0_block, eta=eta, T=T, R=R_box, title_prefix=f"Block Bilinear f(x,y)=x y, eta={eta}")
