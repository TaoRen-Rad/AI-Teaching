import argparse
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import matplotlib.patheffects as pe
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


PALETTE = {
    "paper": "#f5efe4",
    "panel": "#fffdf8",
    "sidebar": "#f1ebdf",
    "ink": "#1f2937",
    "muted": "#6b7280",
    "grid": "#d8dee6",
    "accent": "#0f766e",
    "accent_soft": "#d9f0ec",
}

SCHEME_STYLES = {
    "Exact": {"color": "#111827", "linewidth": 2.8, "linestyle": "-"},
    "Lax-Friedrichs": {"color": "#0f766e", "linewidth": 2.0, "linestyle": "-"},
    "Lax-Wendroff": {"color": "#c2410c", "linewidth": 2.0, "linestyle": "-"},
    "MacCormack": {"color": "#2563eb", "linewidth": 2.0, "linestyle": "-"},
    "Roe": {"color": "#b42318", "linewidth": 2.1, "linestyle": "-"},
}

FIELD_META = {
    "rho": {"title": "Density", "ylabel": r"$\rho$ [kg/m$^3$]"},
    "u": {"title": "Velocity", "ylabel": r"$u$ [m/s]"},
    "p": {"title": "Pressure", "ylabel": r"$p$ [Pa]"},
    "E": {"title": "Specific Total Energy", "ylabel": r"$E$ [J/kg]"},
}

NUMERIC_SCHEMES = ("Lax-Friedrichs", "Lax-Wendroff", "MacCormack", "Roe")


@dataclass
class ShockTubeCase:
    gamma: float = 1.4
    rhoL: float = 1.0
    uL: float = 0.0
    pL: float = 1.0
    rhoR: float = 0.125
    uR: float = 0.0
    pR: float = 0.1
    x0: float = 0.5
    x_min: float = 0.0
    x_max: float = 1.0
    t_end: float = 0.2


@dataclass
class SimulationConfig:
    nx: int = 240
    frames: int = 100
    cfl: float = 0.72


@dataclass
class ExactRiemannSolution:
    gamma: float
    x0: float
    rhoL: float
    uL: float
    pL: float
    rhoR: float
    uR: float
    pR: float
    p_star: float
    u_star: float
    aL: float
    aR: float
    left_wave: str
    right_wave: str
    rho_star_L: float
    rho_star_R: float
    a_star_L: float = None
    a_star_R: float = None
    SL: float = None
    SR: float = None
    SHL: float = None
    STL: float = None
    SHR: float = None
    STR: float = None


def configure_matplotlib_defaults():
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.58,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def sound_speed(gamma, p, rho):
    return np.sqrt(gamma * p / rho)


def pressure_function(p, rho_k, p_k, gamma):
    if p > p_k:
        A = 2.0 / ((gamma + 1.0) * rho_k)
        B = (gamma - 1.0) / (gamma + 1.0) * p_k
        sqrt_term = np.sqrt(A / (p + B))
        f = (p - p_k) * sqrt_term
        df = sqrt_term * (1.0 - 0.5 * (p - p_k) / (p + B))
    else:
        a_k = sound_speed(gamma, p_k, rho_k)
        expo = (gamma - 1.0) / (2.0 * gamma)
        f = (2.0 * a_k / (gamma - 1.0)) * ((p / p_k) ** expo - 1.0)
        df = (1.0 / (rho_k * a_k)) * (p / p_k) ** (
            -(gamma + 1.0) / (2.0 * gamma)
        )
    return f, df


def star_pressure_velocity(rhoL, uL, pL, rhoR, uR, pR, gamma, tol=1e-10, max_iter=100):
    aL = sound_speed(gamma, pL, rhoL)
    aR = sound_speed(gamma, pR, rhoR)
    p_guess = 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (aL + aR)
    p = max(tol, p_guess)

    for _ in range(max_iter):
        fL, dfL = pressure_function(p, rhoL, pL, gamma)
        fR, dfR = pressure_function(p, rhoR, pR, gamma)
        f = fL + fR + (uR - uL)
        df = dfL + dfR
        p_new = max(tol, p - f / df)
        if abs(p_new - p) / max(0.5 * (p_new + p), tol) < tol:
            p = p_new
            break
        p = p_new

    fL, _ = pressure_function(p, rhoL, pL, gamma)
    fR, _ = pressure_function(p, rhoR, pR, gamma)
    u_star = 0.5 * (uL + uR) + 0.5 * (fR - fL)
    return p, u_star


def prepare_exact_solution(case):
    p_star, u_star = star_pressure_velocity(
        case.rhoL, case.uL, case.pL, case.rhoR, case.uR, case.pR, case.gamma
    )
    aL = sound_speed(case.gamma, case.pL, case.rhoL)
    aR = sound_speed(case.gamma, case.pR, case.rhoR)

    left_wave = "shock" if p_star > case.pL else "rarefaction"
    right_wave = "shock" if p_star > case.pR else "rarefaction"

    if left_wave == "shock":
        rho_star_L = case.rhoL * (
            (p_star / case.pL + (case.gamma - 1.0) / (case.gamma + 1.0))
            / ((case.gamma - 1.0) / (case.gamma + 1.0) * p_star / case.pL + 1.0)
        )
        a_star_L = None
        SHL = None
        STL = None
        SL = case.uL - aL * np.sqrt(
            (case.gamma + 1.0) / (2.0 * case.gamma) * (p_star / case.pL)
            + (case.gamma - 1.0) / (2.0 * case.gamma)
        )
    else:
        rho_star_L = case.rhoL * (p_star / case.pL) ** (1.0 / case.gamma)
        a_star_L = aL * (p_star / case.pL) ** ((case.gamma - 1.0) / (2.0 * case.gamma))
        SHL = case.uL - aL
        STL = u_star - a_star_L
        SL = None

    if right_wave == "shock":
        rho_star_R = case.rhoR * (
            (p_star / case.pR + (case.gamma - 1.0) / (case.gamma + 1.0))
            / ((case.gamma - 1.0) / (case.gamma + 1.0) * p_star / case.pR + 1.0)
        )
        a_star_R = None
        SHR = None
        STR = None
        SR = case.uR + aR * np.sqrt(
            (case.gamma + 1.0) / (2.0 * case.gamma) * (p_star / case.pR)
            + (case.gamma - 1.0) / (2.0 * case.gamma)
        )
    else:
        rho_star_R = case.rhoR * (p_star / case.pR) ** (1.0 / case.gamma)
        a_star_R = aR * (p_star / case.pR) ** ((case.gamma - 1.0) / (2.0 * case.gamma))
        SHR = case.uR + aR
        STR = u_star + a_star_R
        SR = None

    return ExactRiemannSolution(
        gamma=case.gamma,
        x0=case.x0,
        rhoL=case.rhoL,
        uL=case.uL,
        pL=case.pL,
        rhoR=case.rhoR,
        uR=case.uR,
        pR=case.pR,
        p_star=p_star,
        u_star=u_star,
        aL=aL,
        aR=aR,
        left_wave=left_wave,
        right_wave=right_wave,
        rho_star_L=rho_star_L,
        rho_star_R=rho_star_R,
        a_star_L=a_star_L,
        a_star_R=a_star_R,
        SL=SL,
        SR=SR,
        SHL=SHL,
        STL=STL,
        SHR=SHR,
        STR=STR,
    )


def sample_exact_solution(x, t, solution):
    if t <= 0.0:
        rho = np.where(x < solution.x0, solution.rhoL, solution.rhoR)
        u = np.where(x < solution.x0, solution.uL, solution.uR)
        p = np.where(x < solution.x0, solution.pL, solution.pR)
        return rho, u, p

    gamma = solution.gamma
    xi = (x - solution.x0) / t
    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)

    mask_left = xi <= solution.u_star
    mask_right = ~mask_left

    if solution.left_wave == "shock":
        mask_far = mask_left & (xi <= solution.SL)
        mask_star = mask_left & ~mask_far
        rho[mask_far], u[mask_far], p[mask_far] = solution.rhoL, solution.uL, solution.pL
        rho[mask_star], u[mask_star], p[mask_star] = (
            solution.rho_star_L,
            solution.u_star,
            solution.p_star,
        )
    else:
        mask_far = mask_left & (xi <= solution.SHL)
        mask_star = mask_left & (xi >= solution.STL)
        mask_fan = mask_left & ~(mask_far | mask_star)
        rho[mask_far], u[mask_far], p[mask_far] = solution.rhoL, solution.uL, solution.pL
        rho[mask_star], u[mask_star], p[mask_star] = (
            solution.rho_star_L,
            solution.u_star,
            solution.p_star,
        )

        if np.any(mask_fan):
            s = xi[mask_fan]
            u_tmp = (2.0 / (gamma + 1.0)) * (
                solution.aL + 0.5 * (gamma - 1.0) * solution.uL + s
            )
            a_tmp = (2.0 / (gamma + 1.0)) * (
                solution.aL + 0.5 * (gamma - 1.0) * (solution.uL - s)
            )
            a_tmp = np.clip(a_tmp, 1e-12, None)
            rho[mask_fan] = solution.rhoL * (a_tmp / solution.aL) ** (
                2.0 / (gamma - 1.0)
            )
            u[mask_fan] = u_tmp
            p[mask_fan] = solution.pL * (a_tmp / solution.aL) ** (
                2.0 * gamma / (gamma - 1.0)
            )

    if solution.right_wave == "shock":
        mask_far = mask_right & (xi >= solution.SR)
        mask_star = mask_right & ~mask_far
        rho[mask_far], u[mask_far], p[mask_far] = solution.rhoR, solution.uR, solution.pR
        rho[mask_star], u[mask_star], p[mask_star] = (
            solution.rho_star_R,
            solution.u_star,
            solution.p_star,
        )
    else:
        mask_far = mask_right & (xi >= solution.SHR)
        mask_star = mask_right & (xi <= solution.STR)
        mask_fan = mask_right & ~(mask_far | mask_star)
        rho[mask_far], u[mask_far], p[mask_far] = solution.rhoR, solution.uR, solution.pR
        rho[mask_star], u[mask_star], p[mask_star] = (
            solution.rho_star_R,
            solution.u_star,
            solution.p_star,
        )

        if np.any(mask_fan):
            s = xi[mask_fan]
            u_tmp = (2.0 / (gamma + 1.0)) * (
                -solution.aR + 0.5 * (gamma - 1.0) * solution.uR + s
            )
            a_tmp = (2.0 / (gamma + 1.0)) * (
                solution.aR - 0.5 * (gamma - 1.0) * (solution.uR - s)
            )
            a_tmp = np.clip(a_tmp, 1e-12, None)
            rho[mask_fan] = solution.rhoR * (a_tmp / solution.aR) ** (
                2.0 / (gamma - 1.0)
            )
            u[mask_fan] = u_tmp
            p[mask_fan] = solution.pR * (a_tmp / solution.aR) ** (
                2.0 * gamma / (gamma - 1.0)
            )

    return rho, u, p


def specific_total_energy(rho, u, p, gamma):
    e = p / ((gamma - 1.0) * rho)
    return e + 0.5 * u**2


def conserved_from_primitive(rho, u, p, gamma):
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return np.stack([rho, rho * u, E], axis=-1)


def primitive_from_conserved(U, gamma):
    rho = np.maximum(U[..., 0], 1e-10)
    u = U[..., 1] / rho
    internal = np.maximum(U[..., 2] - 0.5 * rho * u**2, 1e-12)
    p = (gamma - 1.0) * internal
    return rho, u, p


def flux(U, gamma):
    rho, u, p = primitive_from_conserved(U, gamma)
    return np.stack(
        [
            rho * u,
            rho * u**2 + p,
            u * (U[..., 2] + p),
        ],
        axis=-1,
    )


def enforce_physical_state(U, gamma, rho_floor=1e-8, p_floor=1e-8):
    rho, u, p = primitive_from_conserved(U, gamma)
    rho = np.maximum(rho, rho_floor)
    p = np.maximum(p, p_floor)
    return conserved_from_primitive(rho, u, p, gamma)


def apply_outflow_bc(U, ng=2):
    U[:ng] = U[ng]
    U[-ng:] = U[-ng - 1]


def initialize_state(case, nx, ng=2):
    dx = (case.x_max - case.x_min) / nx
    x = case.x_min + (np.arange(nx) + 0.5) * dx

    rho = np.where(x < case.x0, case.rhoL, case.rhoR)
    u = np.where(x < case.x0, case.uL, case.uR)
    p = np.where(x < case.x0, case.pL, case.pR)

    U_phys = conserved_from_primitive(rho, u, p, case.gamma)
    U = np.zeros((nx + 2 * ng, 3), dtype=float)
    U[ng:-ng] = U_phys
    apply_outflow_bc(U, ng=ng)
    return x, dx, U


def max_signal_speed(U, gamma, ng=2):
    rho, u, p = primitive_from_conserved(U[ng:-ng], gamma)
    a = sound_speed(gamma, p, rho)
    return np.max(np.abs(u) + a)


def step_lax_friedrichs(U, dt, dx, gamma):
    lam = dt / dx
    F = flux(U, gamma)
    U_new = U.copy()
    U_new[2:-2] = 0.5 * (U[3:-1] + U[1:-3]) - 0.5 * lam * (F[3:-1] - F[1:-3])
    apply_outflow_bc(U_new)
    return enforce_physical_state(U_new, gamma)


def step_lax_wendroff(U, dt, dx, gamma):
    lam = dt / dx
    F = flux(U, gamma)
    U_half = 0.5 * (U[:-1] + U[1:]) - 0.5 * lam * (F[1:] - F[:-1])
    U_half = enforce_physical_state(U_half, gamma)
    F_half = flux(U_half, gamma)

    U_new = U.copy()
    U_new[2:-2] = U[2:-2] - lam * (F_half[2:-1] - F_half[1:-2])
    apply_outflow_bc(U_new)
    return enforce_physical_state(U_new, gamma)


def add_adaptive_viscosity(U, gamma, strength=0.22):
    rho, u, p = primitive_from_conserved(U, gamma)
    sensor = np.zeros_like(rho)
    numerator = np.abs(p[2:] - 2.0 * p[1:-1] + p[:-2])
    denominator = np.abs(p[2:] + 2.0 * p[1:-1] + p[:-2]) + 1e-12
    sensor[1:-1] = strength * numerator / denominator

    U_smooth = U.copy()
    U_smooth[2:-2] += (
        sensor[2:-2, None] * (U[3:-1] - U[2:-2])
        - sensor[1:-3, None] * (U[2:-2] - U[1:-3])
    )
    apply_outflow_bc(U_smooth)
    return U_smooth


def step_maccormack(U, dt, dx, gamma):
    lam = dt / dx
    F = flux(U, gamma)

    U_pred = U.copy()
    U_pred[2:-2] = U[2:-2] - lam * (F[3:-1] - F[2:-2])
    apply_outflow_bc(U_pred)
    U_pred = enforce_physical_state(U_pred, gamma)
    F_pred = flux(U_pred, gamma)

    U_new = U.copy()
    U_new[2:-2] = 0.5 * (
        U[2:-2] + U_pred[2:-2] - lam * (F_pred[2:-2] - F_pred[1:-3])
    )
    U_new = add_adaptive_viscosity(U_new, gamma)
    apply_outflow_bc(U_new)
    return enforce_physical_state(U_new, gamma)


def roe_flux(U_L, U_R, gamma):
    rhoL, uL, pL = primitive_from_conserved(U_L, gamma)
    rhoR, uR, pR = primitive_from_conserved(U_R, gamma)
    HL = (U_L[..., 2] + pL) / rhoL
    HR = (U_R[..., 2] + pR) / rhoR

    sqrtL = np.sqrt(rhoL)
    sqrtR = np.sqrt(rhoR)
    denom = np.maximum(sqrtL + sqrtR, 1e-12)

    u_tilde = (sqrtL * uL + sqrtR * uR) / denom
    H_tilde = (sqrtL * HL + sqrtR * HR) / denom
    a_tilde = np.sqrt(np.maximum((gamma - 1.0) * (H_tilde - 0.5 * u_tilde**2), 1e-12))
    rho_tilde = sqrtL * sqrtR

    drho = rhoR - rhoL
    du = uR - uL
    dp = pR - pL

    alpha2 = drho - dp / np.maximum(a_tilde**2, 1e-12)
    alpha1 = (dp - rho_tilde * a_tilde * du) / np.maximum(2.0 * a_tilde**2, 1e-12)
    alpha3 = (dp + rho_tilde * a_tilde * du) / np.maximum(2.0 * a_tilde**2, 1e-12)

    lambdas = np.stack([u_tilde - a_tilde, u_tilde, u_tilde + a_tilde], axis=-1)
    delta = 0.12 * a_tilde[..., None]
    abs_lambdas = np.abs(lambdas)
    entropy_mask = abs_lambdas < delta
    abs_lambdas = np.where(
        entropy_mask,
        0.5 * (abs_lambdas**2 / np.maximum(delta, 1e-12) + delta),
        abs_lambdas,
    )

    r1 = np.stack(
        [np.ones_like(u_tilde), u_tilde - a_tilde, H_tilde - u_tilde * a_tilde],
        axis=-1,
    )
    r2 = np.stack(
        [np.ones_like(u_tilde), u_tilde, 0.5 * u_tilde**2],
        axis=-1,
    )
    r3 = np.stack(
        [np.ones_like(u_tilde), u_tilde + a_tilde, H_tilde + u_tilde * a_tilde],
        axis=-1,
    )

    dissipation = (
        abs_lambdas[..., 0, None] * alpha1[..., None] * r1
        + abs_lambdas[..., 1, None] * alpha2[..., None] * r2
        + abs_lambdas[..., 2, None] * alpha3[..., None] * r3
    )

    return 0.5 * (flux(U_L, gamma) + flux(U_R, gamma)) - 0.5 * dissipation


def step_roe(U, dt, dx, gamma):
    lam = dt / dx
    fluxes = roe_flux(U[:-1], U[1:], gamma)
    U_new = U.copy()
    U_new[2:-2] = U[2:-2] - lam * (fluxes[2:-1] - fluxes[1:-2])
    apply_outflow_bc(U_new)
    return enforce_physical_state(U_new, gamma)


STEP_FUNCTIONS = {
    "Lax-Friedrichs": step_lax_friedrichs,
    "Lax-Wendroff": step_lax_wendroff,
    "MacCormack": step_maccormack,
    "Roe": step_roe,
}


def physical_fields_from_state(U, gamma, ng=2):
    rho, u, p = primitive_from_conserved(U[ng:-ng], gamma)
    E = specific_total_energy(rho, u, p, gamma)
    return rho, u, p, E


def solve_numerical_scheme(case, config, scheme_name, output_times, progress_callback=None):
    x, dx, U = initialize_state(case, config.nx)
    snapshots = {field: np.zeros((len(output_times), config.nx), dtype=float) for field in FIELD_META}

    rho, u, p, E = physical_fields_from_state(U, case.gamma)
    snapshots["rho"][0] = rho
    snapshots["u"][0] = u
    snapshots["p"][0] = p
    snapshots["E"][0] = E

    step_fn = STEP_FUNCTIONS[scheme_name]
    t = 0.0
    frame_index = 0

    while frame_index < len(output_times) - 1:
        target_time = output_times[frame_index + 1]
        while t < target_time - 1e-12:
            apply_outflow_bc(U)
            s_max = max_signal_speed(U, case.gamma)
            dt = min(config.cfl * dx / max(s_max, 1e-12), target_time - t)
            U = step_fn(U, dt, dx, case.gamma)
            t += dt
            if progress_callback is not None:
                progress_callback(min(t / case.t_end, 1.0))

        frame_index += 1
        rho, u, p, E = physical_fields_from_state(U, case.gamma)
        snapshots["rho"][frame_index] = rho
        snapshots["u"][frame_index] = u
        snapshots["p"][frame_index] = p
        snapshots["E"][frame_index] = E

    return snapshots


def compute_exact_snapshots(case, x, output_times):
    solution = prepare_exact_solution(case)
    snapshots = {field: np.zeros((len(output_times), x.size), dtype=float) for field in FIELD_META}

    for idx, t in enumerate(output_times):
        rho, u, p = sample_exact_solution(x, t, solution)
        snapshots["rho"][idx] = rho
        snapshots["u"][idx] = u
        snapshots["p"][idx] = p
        snapshots["E"][idx] = specific_total_energy(rho, u, p, case.gamma)

    return snapshots


def compute_density_errors(numerical_snapshots, exact_snapshots):
    diff = numerical_snapshots["rho"][-1] - exact_snapshots["rho"][-1]
    return {
        "L1(rho)": float(np.mean(np.abs(diff))),
        "Linf(rho)": float(np.max(np.abs(diff))),
    }


def precompute_all(case, config, progress_callback=None):
    x = case.x_min + (np.arange(config.nx) + 0.5) * (case.x_max - case.x_min) / config.nx
    output_times = np.linspace(0.0, case.t_end, config.frames)
    results = {}
    metrics = {}

    if progress_callback is not None:
        progress_callback("Sampling exact solution...", 0.02)
    exact_snapshots = compute_exact_snapshots(case, x, output_times)
    results["Exact"] = exact_snapshots

    total_numeric = len(NUMERIC_SCHEMES)
    for idx, scheme_name in enumerate(NUMERIC_SCHEMES):
        base = 0.08 + idx * (0.92 / total_numeric)
        span = 0.92 / total_numeric

        def scheme_progress(local_fraction, scheme=scheme_name, base_fraction=base, span_fraction=span):
            if progress_callback is not None:
                progress_callback(
                    f"Computing {scheme}...",
                    min(base_fraction + span_fraction * local_fraction, 0.995),
                )

        numeric_snapshots = solve_numerical_scheme(
            case,
            config,
            scheme_name,
            output_times,
            progress_callback=scheme_progress,
        )
        results[scheme_name] = numeric_snapshots
        metrics[scheme_name] = compute_density_errors(numeric_snapshots, exact_snapshots)

    if progress_callback is not None:
        progress_callback("Ready to animate.", 1.0)

    return {
        "x": x,
        "times": output_times,
        "results": results,
        "metrics": metrics,
        "case": case,
        "config": config,
    }


def padded_limits(values):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    pad = 0.08 * span if span > 1e-12 else 0.1 * max(abs(vmin), 1.0)
    return vmin - pad, vmax + pad


def build_plot_limits(results_bundle):
    limits = {}
    for field in FIELD_META:
        stacked = np.concatenate(
            [results_bundle["results"][scheme][field] for scheme in results_bundle["results"]],
            axis=0,
        )
        limits[field] = padded_limits(stacked)
    return limits


def create_static_preview(results_bundle, output_path):
    limits = build_plot_limits(results_bundle)
    x = results_bundle["x"]
    final_index = -1

    figure = Figure(figsize=(14.0, 8.8), facecolor=PALETTE["paper"])
    axes_arr = figure.subplots(2, 2)
    axes = {
        "rho": axes_arr[0, 0],
        "u": axes_arr[0, 1],
        "p": axes_arr[1, 0],
        "E": axes_arr[1, 1],
    }

    figure.subplots_adjust(top=0.790, left=0.08, right=0.98, bottom=0.10, hspace=0.30, wspace=0.22)
    figure.text(0.08, 0.965, "Shock Tube Scheme Comparison", fontsize=19, fontweight="bold", color=PALETTE["ink"])
    figure.text(
        0.08,
        0.932,
        "Exact solution and four numerical schemes evaluated on a shared time sequence.",
        fontsize=10.2,
        color=PALETTE["muted"],
    )
    figure.text(
        0.98,
        0.965,
        f"t = {results_bundle['times'][final_index]:.3f} s",
        ha="right",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f7f3eb", "edgecolor": "#d6d3d1", "linewidth": 0.9},
    )

    progress_bg = Rectangle((0.08, 0.900), 0.90, 0.007, transform=figure.transFigure, linewidth=0, facecolor="#d8e5e2", zorder=20)
    progress_bar = Rectangle((0.08, 0.900), 0.90, 0.007, transform=figure.transFigure, linewidth=0, facecolor=PALETTE["accent"], zorder=21)
    figure.add_artist(progress_bg)
    figure.add_artist(progress_bar)

    legend_handles = []
    legend_labels = []
    for field, ax in axes.items():
        ax.set_facecolor(PALETTE["panel"])
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(*limits[field])
        ax.set_title(FIELD_META[field]["title"], loc="left", fontweight="bold", pad=8)
        ax.set_ylabel(FIELD_META[field]["ylabel"])
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.55)
        ax.spines["left"].set_alpha(0.55)
        ax.spines["bottom"].set_alpha(0.55)

        for scheme_name, style in SCHEME_STYLES.items():
            line, = ax.plot(
                x,
                results_bundle["results"][scheme_name][field][final_index],
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                label=scheme_name,
            )
            if scheme_name == "Exact":
                line.set_path_effects(
                    [pe.Stroke(linewidth=4.4, foreground="white", alpha=0.82), pe.Normal()]
                )
            if field == "rho":
                legend_handles.append(line)
                legend_labels.append(scheme_name)

    axes["p"].set_xlabel("x [m]")
    axes["E"].set_xlabel("x [m]")
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.54, 0.852),
        prop={"size": 10},
        handlelength=1.8,
        columnspacing=1.8,
    )
    figure.savefig(output_path, dpi=150, facecolor=figure.get_facecolor())


class ShockTubeComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shock Tube Scheme Comparison Studio")
        self.root.geometry("1580x960")
        self.root.minsize(1380, 860)
        self.root.configure(bg=PALETTE["paper"])

        self.results_bundle = None
        self.line_artists = {}
        self.plot_limits = None
        self.current_frame = 0
        self.playing = False
        self.after_id = None

        self.gamma_var = tk.DoubleVar(value=1.4)
        self.t_end_var = tk.DoubleVar(value=0.2)
        self.cfl_var = tk.DoubleVar(value=0.72)
        self.nx_var = tk.IntVar(value=240)
        self.frames_var = tk.IntVar(value=100)
        self.xmin_var = tk.DoubleVar(value=0.0)
        self.xmax_var = tk.DoubleVar(value=1.0)
        self.x0_var = tk.DoubleVar(value=0.5)

        self.rhoL_var = tk.DoubleVar(value=1.0)
        self.uL_var = tk.DoubleVar(value=0.0)
        self.pL_var = tk.DoubleVar(value=1.0)
        self.rhoR_var = tk.DoubleVar(value=0.125)
        self.uR_var = tk.DoubleVar(value=0.0)
        self.pR_var = tk.DoubleVar(value=0.1)

        self.speed_var = tk.DoubleVar(value=1.0)
        self.frame_slider_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Ready. Configure the case and click Compute All.")
        self.time_var = tk.StringVar(value="t = 0.0000 / 0.0000 s")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.visibility_vars = {
            scheme_name: tk.BooleanVar(value=True)
            for scheme_name in SCHEME_STYLES
        }

        self._configure_styles()
        self._build_ui()
        self._build_figure()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Main.TFrame", background=PALETTE["paper"])
        style.configure("Sidebar.TFrame", background=PALETTE["sidebar"])
        style.configure("Panel.TLabelframe", background=PALETTE["sidebar"], foreground=PALETTE["ink"])
        style.configure(
            "Panel.TLabelframe.Label",
            background=PALETTE["sidebar"],
            foreground=PALETTE["ink"],
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("Sidebar.TLabel", background=PALETTE["sidebar"], foreground=PALETTE["ink"])
        style.configure("Status.TLabel", background=PALETTE["sidebar"], foreground=PALETTE["muted"])
        style.configure("Accent.TButton", background=PALETTE["accent"], foreground="white", borderwidth=0, padding=(10, 6))
        style.map(
            "Accent.TButton",
            background=[("active", "#0b5c56"), ("disabled", "#7aa8a2")],
            foreground=[("disabled", "#ecfdf5")],
        )
        style.configure("Soft.TButton", background="#e8edf2", foreground=PALETTE["ink"], padding=(10, 6))
        style.configure("TCheckbutton", background=PALETTE["sidebar"], foreground=PALETTE["ink"])
        style.configure("TScale", background=PALETTE["paper"])
        style.configure(
            "Compute.Horizontal.TProgressbar",
            troughcolor="#d9e5e5",
            background=PALETTE["accent"],
            bordercolor="#d9e5e5",
            lightcolor=PALETTE["accent"],
            darkcolor=PALETTE["accent"],
        )
        style.configure(
            "Metrics.Treeview",
            rowheight=24,
            fieldbackground=PALETTE["panel"],
            background=PALETTE["panel"],
            foreground=PALETTE["ink"],
        )
        style.configure("Metrics.Treeview.Heading", background="#e8edf2", foreground=PALETTE["ink"])

    def _build_ui(self):
        self.main_frame = ttk.Frame(self.root, style="Main.TFrame")
        self.main_frame.pack(fill="both", expand=True)

        self.sidebar_container = ttk.Frame(self.main_frame, style="Sidebar.TFrame", width=360)
        self.sidebar_container.pack(side="left", fill="y")
        self.sidebar_container.pack_propagate(False)

        self.sidebar_canvas = tk.Canvas(
            self.sidebar_container,
            bg=PALETTE["sidebar"],
            highlightthickness=0,
            bd=0,
        )
        self.sidebar_scrollbar = ttk.Scrollbar(
            self.sidebar_container,
            orient="vertical",
            command=self.sidebar_canvas.yview,
        )
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        self.sidebar_scrollbar.pack(side="right", fill="y")

        self.sidebar = ttk.Frame(self.sidebar_canvas, style="Sidebar.TFrame")
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
        self.sidebar.bind("<Configure>", self._on_sidebar_frame_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)
        self.sidebar_canvas.bind("<Enter>", self._bind_sidebar_mousewheel)
        self.sidebar_canvas.bind("<Leave>", self._unbind_sidebar_mousewheel)

        self.content = ttk.Frame(self.main_frame, style="Main.TFrame")
        self.content.pack(side="right", fill="both", expand=True, padx=(14, 14), pady=(12, 12))

        title = ttk.Label(
            self.sidebar,
            text="Shock Tube\nComparison Studio",
            style="Sidebar.TLabel",
            font=("Segoe UI", 18, "bold"),
            justify="left",
        )
        title.pack(anchor="w", padx=16, pady=(18, 6))

        subtitle = ttk.Label(
            self.sidebar,
            text="Precompute exact and numerical solutions, then inspect them as an animation.",
            style="Status.TLabel",
            wraplength=290,
            justify="left",
        )
        subtitle.pack(anchor="w", padx=16, pady=(0, 14))

        self._build_case_panel()
        self._build_state_panels()
        self._build_scheme_panel()
        self._build_action_panel()
        self._build_metrics_panel()

        footer = ttk.Label(
            self.sidebar,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=300,
            justify="left",
        )
        footer.pack(fill="x", padx=16, pady=(10, 12))

        self.figure_frame = ttk.Frame(self.content, style="Main.TFrame")
        self.figure_frame.pack(fill="both", expand=True)

        self.playback_frame = ttk.Frame(self.content, style="Main.TFrame")
        self.playback_frame.pack(fill="x", pady=(8, 0))

        self.play_button = ttk.Button(self.playback_frame, text="Play", style="Soft.TButton", command=self.toggle_playback)
        self.play_button.pack(side="left")
        self.reset_button = ttk.Button(self.playback_frame, text="Reset", style="Soft.TButton", command=self.reset_playback)
        self.reset_button.pack(side="left", padx=(8, 10))

        ttk.Label(self.playback_frame, text="Playback speed", style="Sidebar.TLabel").pack(side="left", padx=(4, 8))
        self.speed_scale = ttk.Scale(self.playback_frame, from_=0.4, to=2.5, variable=self.speed_var, orient="horizontal")
        self.speed_scale.pack(side="left", fill="x", expand=False, ipadx=50)

        ttk.Label(self.playback_frame, textvariable=self.time_var, style="Sidebar.TLabel").pack(side="right")

        self.frame_scale = ttk.Scale(
            self.content,
            from_=0,
            to=1,
            orient="horizontal",
            variable=self.frame_slider_var,
            command=self._on_slider_changed,
        )
        self.frame_scale.pack(fill="x", pady=(10, 0))

    def _on_sidebar_frame_configure(self, _event):
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_sidebar_canvas_configure(self, event):
        self.sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width)

    def _bind_sidebar_mousewheel(self, _event):
        self.sidebar_canvas.bind_all("<MouseWheel>", self._on_sidebar_mousewheel)

    def _unbind_sidebar_mousewheel(self, _event):
        self.sidebar_canvas.unbind_all("<MouseWheel>")

    def _on_sidebar_mousewheel(self, event):
        self.sidebar_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _build_case_panel(self):
        panel = ttk.LabelFrame(self.sidebar, text="Case Setup", style="Panel.TLabelframe")
        panel.pack(fill="x", padx=16, pady=(0, 10))

        entries = [
            ("gamma", self.gamma_var),
            ("t_end [s]", self.t_end_var),
            ("CFL", self.cfl_var),
            ("Nx", self.nx_var),
            ("frames", self.frames_var),
            ("x_min [m]", self.xmin_var),
            ("x_max [m]", self.xmax_var),
            ("x0 [m]", self.x0_var),
        ]

        for idx, (label, var) in enumerate(entries):
            ttk.Label(panel, text=label, style="Sidebar.TLabel").grid(row=idx, column=0, sticky="w", padx=10, pady=4)
            entry = ttk.Entry(panel, textvariable=var, width=12)
            entry.grid(row=idx, column=1, sticky="ew", padx=10, pady=4)

        panel.columnconfigure(1, weight=1)

    def _build_state_panels(self):
        left_panel = ttk.LabelFrame(self.sidebar, text="Left State", style="Panel.TLabelframe")
        left_panel.pack(fill="x", padx=16, pady=(0, 10))
        right_panel = ttk.LabelFrame(self.sidebar, text="Right State", style="Panel.TLabelframe")
        right_panel.pack(fill="x", padx=16, pady=(0, 10))

        for idx, (label, var) in enumerate(
            [("rho [kg/m^3]", self.rhoL_var), ("u [m/s]", self.uL_var), ("p [Pa]", self.pL_var)]
        ):
            ttk.Label(left_panel, text=label, style="Sidebar.TLabel").grid(row=idx, column=0, sticky="w", padx=10, pady=4)
            ttk.Entry(left_panel, textvariable=var, width=12).grid(row=idx, column=1, sticky="ew", padx=10, pady=4)
        left_panel.columnconfigure(1, weight=1)

        for idx, (label, var) in enumerate(
            [("rho [kg/m^3]", self.rhoR_var), ("u [m/s]", self.uR_var), ("p [Pa]", self.pR_var)]
        ):
            ttk.Label(right_panel, text=label, style="Sidebar.TLabel").grid(row=idx, column=0, sticky="w", padx=10, pady=4)
            ttk.Entry(right_panel, textvariable=var, width=12).grid(row=idx, column=1, sticky="ew", padx=10, pady=4)
        right_panel.columnconfigure(1, weight=1)

    def _build_scheme_panel(self):
        panel = ttk.LabelFrame(self.sidebar, text="Visible Curves", style="Panel.TLabelframe")
        panel.pack(fill="x", padx=16, pady=(0, 10))

        for idx, scheme_name in enumerate(SCHEME_STYLES):
            ttk.Checkbutton(
                panel,
                text=scheme_name,
                variable=self.visibility_vars[scheme_name],
                command=self.refresh_visibility,
            ).grid(row=idx, column=0, sticky="w", padx=10, pady=2)

    def _build_action_panel(self):
        panel = ttk.LabelFrame(self.sidebar, text="Actions", style="Panel.TLabelframe")
        panel.pack(fill="x", padx=16, pady=(0, 10))

        self.compute_button = ttk.Button(panel, text="Compute All Schemes", style="Accent.TButton", command=self.compute_all)
        self.compute_button.pack(fill="x", padx=10, pady=(10, 8))

        self.progress_bar = ttk.Progressbar(
            panel,
            orient="horizontal",
            mode="determinate",
            variable=self.progress_var,
            maximum=100.0,
            style="Compute.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))

    def _build_metrics_panel(self):
        panel = ttk.LabelFrame(self.sidebar, text="Final Density Error", style="Panel.TLabelframe")
        panel.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        self.metrics_tree = ttk.Treeview(
            panel,
            columns=("scheme", "l1", "linf"),
            show="headings",
            height=6,
            style="Metrics.Treeview",
        )
        self.metrics_tree.heading("scheme", text="Scheme")
        self.metrics_tree.heading("l1", text="L1")
        self.metrics_tree.heading("linf", text="Linf")
        self.metrics_tree.column("scheme", width=110, anchor="w")
        self.metrics_tree.column("l1", width=80, anchor="center")
        self.metrics_tree.column("linf", width=80, anchor="center")
        self.metrics_tree.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_figure(self):
        self.figure = Figure(figsize=(11.2, 7.8), facecolor=PALETTE["paper"])
        grid = self.figure.add_gridspec(
            2,
            2,
            left=0.06,
            right=0.98,
            top=0.79,
            bottom=0.10,
            hspace=0.30,
            wspace=0.20,
        )
        self.axes = {
            "rho": self.figure.add_subplot(grid[0, 0]),
            "u": self.figure.add_subplot(grid[0, 1]),
            "p": self.figure.add_subplot(grid[1, 0]),
            "E": self.figure.add_subplot(grid[1, 1]),
        }

        self.figure.text(0.06, 0.965, "Shock Tube Numerical vs Exact Comparison", fontsize=18.5, fontweight="bold", color=PALETTE["ink"])
        self.figure.text(
            0.06,
            0.932,
            "Exact solution and four numerical schemes compared frame by frame on a shared timeline.",
            fontsize=10.0,
            color=PALETTE["muted"],
        )
        self.fig_time_text = self.figure.text(
            0.98,
            0.965,
            "t = 0.0000 / 0.0000 s",
            ha="right",
            fontsize=12.8,
            fontweight="bold",
            color=PALETTE["ink"],
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f7f3eb", "edgecolor": "#d6d3d1", "linewidth": 0.9},
        )

        self.progress_bg = Rectangle((0.06, 0.900), 0.90, 0.007, transform=self.figure.transFigure, linewidth=0, facecolor="#d8e5e2", zorder=20)
        self.progress_bar_patch = Rectangle((0.06, 0.900), 0.002, 0.007, transform=self.figure.transFigure, linewidth=0, facecolor=PALETTE["accent"], zorder=21)
        self.figure.add_artist(self.progress_bg)
        self.figure.add_artist(self.progress_bar_patch)

        self.line_artists = {field: {} for field in FIELD_META}
        legend_handles = []
        legend_labels = []
        x_placeholder = np.linspace(0.0, 1.0, 10)

        for field, ax in self.axes.items():
            ax.set_facecolor(PALETTE["panel"])
            ax.set_title(FIELD_META[field]["title"], loc="left", fontweight="bold", pad=8)
            ax.set_ylabel(FIELD_META[field]["ylabel"])
            ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.55)
            ax.spines["left"].set_alpha(0.55)
            ax.spines["bottom"].set_alpha(0.55)
            ax.set_xlim(0.0, 1.0)

            for scheme_name, style in SCHEME_STYLES.items():
                line, = ax.plot(
                    x_placeholder,
                    np.zeros_like(x_placeholder),
                    color=style["color"],
                    linewidth=style["linewidth"],
                    linestyle=style["linestyle"],
                    label=scheme_name,
                    alpha=1.0 if scheme_name == "Exact" else 0.95,
                )
                if scheme_name == "Exact":
                    line.set_path_effects(
                        [pe.Stroke(linewidth=4.2, foreground="white", alpha=0.82), pe.Normal()]
                    )
                self.line_artists[field][scheme_name] = line
                if field == "rho":
                    legend_handles.append(line)
                    legend_labels.append(scheme_name)

        self.axes["p"].set_xlabel("x [m]")
        self.axes["E"].set_xlabel("x [m]")
        self.figure.legend(
            legend_handles,
            legend_labels,
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.852),
            frameon=False,
            prop={"size": 10},
            handlelength=1.8,
            columnspacing=1.8,
        )

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _read_case_from_controls(self):
        case = ShockTubeCase(
            gamma=float(self.gamma_var.get()),
            rhoL=float(self.rhoL_var.get()),
            uL=float(self.uL_var.get()),
            pL=float(self.pL_var.get()),
            rhoR=float(self.rhoR_var.get()),
            uR=float(self.uR_var.get()),
            pR=float(self.pR_var.get()),
            x0=float(self.x0_var.get()),
            x_min=float(self.xmin_var.get()),
            x_max=float(self.xmax_var.get()),
            t_end=float(self.t_end_var.get()),
        )
        config = SimulationConfig(
            nx=int(self.nx_var.get()),
            frames=int(self.frames_var.get()),
            cfl=float(self.cfl_var.get()),
        )

        if case.x_max <= case.x_min:
            raise ValueError("x_max must be larger than x_min.")
        if not (case.x_min < case.x0 < case.x_max):
            raise ValueError("x0 must stay inside the domain.")
        if case.t_end <= 0.0:
            raise ValueError("t_end must be positive.")
        if config.nx < 80:
            raise ValueError("Nx should be at least 80 for a meaningful comparison.")
        if config.frames < 10:
            raise ValueError("frames should be at least 10.")
        if config.cfl <= 0.0:
            raise ValueError("CFL must be positive.")

        return case, config

    def _set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def _set_compute_state(self, is_busy):
        state = "disabled" if is_busy else "normal"
        self.compute_button.configure(state=state)
        self.play_button.configure(state=state if self.results_bundle is None else "normal")
        self.reset_button.configure(state=state if self.results_bundle is None else "normal")

    def compute_all(self):
        try:
            case, config = self._read_case_from_controls()
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self.stop_playback()
        self._set_compute_state(True)
        self.progress_var.set(0.0)
        self._set_status("Starting precomputation...")

        def progress_callback(message, fraction):
            self.progress_var.set(100.0 * fraction)
            self._set_status(message)

        try:
            self.results_bundle = precompute_all(case, config, progress_callback=progress_callback)
        except Exception as exc:
            self._set_compute_state(False)
            messagebox.showerror("Computation Failed", str(exc))
            return

        self._populate_metrics()
        self._configure_axes_from_results()
        self.current_frame = 0
        self.frame_slider_var.set(0.0)
        self.frame_scale.configure(to=max(len(self.results_bundle["times"]) - 1, 1))
        self.update_frame(0)
        self._set_status("Computed all schemes. Use Play or drag the slider to inspect the evolution.")
        self._set_compute_state(False)
        self.play_button.configure(state="normal", text="Play")
        self.reset_button.configure(state="normal")

    def _populate_metrics(self):
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)

        for scheme_name in NUMERIC_SCHEMES:
            metrics = self.results_bundle["metrics"][scheme_name]
            self.metrics_tree.insert(
                "",
                "end",
                values=(
                    scheme_name,
                    f"{metrics['L1(rho)']:.4e}",
                    f"{metrics['Linf(rho)']:.4e}",
                ),
            )

    def _configure_axes_from_results(self):
        self.plot_limits = build_plot_limits(self.results_bundle)
        x = self.results_bundle["x"]

        for field, ax in self.axes.items():
            ax.set_xlim(float(x[0]), float(x[-1]))
            ax.set_ylim(*self.plot_limits[field])

    def refresh_visibility(self):
        for field in FIELD_META:
            for scheme_name, line in self.line_artists[field].items():
                line.set_visible(self.visibility_vars[scheme_name].get())
        self.canvas.draw_idle()

    def update_frame(self, frame_index):
        if self.results_bundle is None:
            return

        frame_index = int(np.clip(frame_index, 0, len(self.results_bundle["times"]) - 1))
        self.current_frame = frame_index
        x = self.results_bundle["x"]
        time_value = self.results_bundle["times"][frame_index]
        t_end = self.results_bundle["times"][-1]
        progress = 1.0 if t_end <= 0.0 else time_value / t_end

        for field in FIELD_META:
            for scheme_name in SCHEME_STYLES:
                self.line_artists[field][scheme_name].set_data(
                    x,
                    self.results_bundle["results"][scheme_name][field][frame_index],
                )
                self.line_artists[field][scheme_name].set_visible(self.visibility_vars[scheme_name].get())

        self.fig_time_text.set_text(f"t = {time_value:.4f} s / {t_end:.4f} s")
        self.progress_bar_patch.set_width(0.90 * max(progress, 0.002))
        self.time_var.set(f"t = {time_value:.4f} / {t_end:.4f} s")
        self.canvas.draw_idle()

    def _on_slider_changed(self, value):
        if self.results_bundle is None:
            return
        self.stop_playback(update_button=False)
        self.play_button.configure(text="Play")
        self.update_frame(int(float(value)))

    def toggle_playback(self):
        if self.results_bundle is None:
            messagebox.showinfo("No Data Yet", "Compute the schemes first, then start playback.")
            return

        if self.playing:
            self.stop_playback()
            self.play_button.configure(text="Play")
        else:
            self.playing = True
            self.play_button.configure(text="Pause")
            self._playback_tick()

    def _playback_tick(self):
        if not self.playing or self.results_bundle is None:
            return

        next_frame = self.current_frame + 1
        if next_frame >= len(self.results_bundle["times"]):
            self.stop_playback(update_button=False)
            self.play_button.configure(text="Play")
            return

        self.frame_slider_var.set(next_frame)
        self.update_frame(next_frame)
        delay_ms = max(18, int(42 / max(self.speed_var.get(), 0.2)))
        self.after_id = self.root.after(delay_ms, self._playback_tick)

    def stop_playback(self, update_button=True):
        self.playing = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if update_button:
            self.play_button.configure(text="Play")

    def reset_playback(self):
        self.stop_playback()
        if self.results_bundle is None:
            return
        self.frame_slider_var.set(0.0)
        self.update_frame(0)


def run_verification(output_path=None):
    configure_matplotlib_defaults()
    case = ShockTubeCase()
    config = SimulationConfig(nx=180, frames=50, cfl=0.68)
    bundle = precompute_all(case, config)

    print("Verification summary:")
    for scheme_name in NUMERIC_SCHEMES:
        metrics = bundle["metrics"][scheme_name]
        print(
            f"{scheme_name:>14s} | "
            f"L1(rho) = {metrics['L1(rho)']:.4e} | "
            f"Linf(rho) = {metrics['Linf(rho)']:.4e}"
        )

    if output_path is not None:
        create_static_preview(bundle, output_path)
        print(f"Preview saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Shock tube comparison UI for exact and numerical schemes."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a non-UI verification computation and print final errors.",
    )
    parser.add_argument(
        "--preview",
        type=str,
        default="shocktube_compare_preview.png",
        help="Preview image path for --verify mode.",
    )
    args = parser.parse_args()

    if args.verify:
        preview_path = Path(args.preview).resolve()
        run_verification(output_path=preview_path)
        return

    configure_matplotlib_defaults()
    root = tk.Tk()
    ShockTubeComparisonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
