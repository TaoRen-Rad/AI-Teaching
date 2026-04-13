from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from io import BytesIO, StringIO
from typing import Any

import matplotlib.cm as cm
import numpy as np
import streamlit as st
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from lid_driven_cavity_streamfunction_vorticity_animation import (
    FlowSnapshot,
    LidDrivenCavitySolver,
    SolverConfig,
)
from lid_driven_cavity_streamfunction_vorticity_thermal_animation import (
    LidDrivenCavityThermalSolver,
    ThermalFlowSnapshot,
    ThermalSolverConfig,
)


PALETTE = {
    "paper": "#f8f6f2",
    "panel": "#ffffff",
    "ink": "#132238",
    "muted": "#5b6b82",
    "accent": "#0f766e",
    "accent_soft": "#d8ece8",
    "secondary": "#c2410c",
    "grid": "#dbe5f0",
}

DEFAULTS: dict[str, Any] = {
    "mode": "flow",
    "nx": 101,
    "ny": 101,
    "reynolds": 400.0,
    "lid_velocity": 1.0,
    "final_time": 10.0,
    "cfl_number": 0.50,
    "frame_stride": 100,
    "poisson_max_iter": 60,
    "thermal_diffusivity": 5.0e-4,
    "thermal_conductivity": 1.0,
    "initial_temperature": 300.0,
    "top_temperature": 300.0,
    "left_temperature": 300.0,
    "right_temperature": 300.0,
    "bottom_bc_mode": "fixed_temperature",
    "bottom_temperature": 320.0,
    "bottom_heat_flux": 0.0,
    "volumetric_heat_source": 0.0,
    "source_x_min": 0.40,
    "source_x_max": 0.60,
    "source_y_min": 0.20,
    "source_y_max": 0.40,
}

PRESETS = {
    "Classical Laminar Cavity": {
        "description": "Baseline lid-driven cavity benchmark for streamlines, streamfunction, vorticity, and pressure recovery.",
        "mode": "flow",
    },
    "Heated Bottom Wall": {
        "description": "Adds the transient temperature equation with a hotter bottom wall so advection and diffusion are easy to compare.",
        "mode": "thermal",
    },
    "Internal Heat Source": {
        "description": "Uses the thermal solver with an interior source region to show how the flow convects a thermal plume.",
        "mode": "thermal",
        "final_time": 8.0,
        "bottom_temperature": 300.0,
        "volumetric_heat_source": 5.0e5,
    },
}


@dataclass
class SimulationRecord:
    mode: str
    config: SolverConfig | ThermalSolverConfig
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    snapshots: list[FlowSnapshot | ThermalFlowSnapshot]
    ranges: dict[str, tuple[float, float]]
    levels: dict[str, np.ndarray]
    history: dict[str, np.ndarray]
    signature: tuple[Any, ...]


def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("preset", "Classical Laminar Cavity")
    st.session_state.setdefault("_last_preset", None)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("frame_index", 0)
    st.session_state.setdefault("view_name", "Overview Fields")


def apply_preset(name: str) -> None:
    values = dict(DEFAULTS)
    values.update(PRESETS[name])
    for key, value in values.items():
        if key != "description":
            st.session_state[key] = value


def safe_range(vmin: float, vmax: float, pad: float = 1.0) -> tuple[float, float]:
    if abs(vmax - vmin) < 1.0e-12:
        center = 0.5 * (vmin + vmax)
        return center - pad, center + pad
    return float(vmin), float(vmax)


def signature(mode: str, config: SolverConfig | ThermalSolverConfig) -> tuple[Any, ...]:
    return mode, tuple(sorted((key, repr(value)) for key, value in asdict(config).items()))


def build_config() -> tuple[str, SolverConfig | ThermalSolverConfig]:
    mode = str(st.session_state["mode"]).strip().lower()
    nx = int(st.session_state["nx"])
    ny = int(st.session_state["ny"])
    reynolds = float(st.session_state["reynolds"])
    lid_velocity = float(st.session_state["lid_velocity"])
    final_time = float(st.session_state["final_time"])
    cfl_number = float(st.session_state["cfl_number"])
    frame_stride = int(st.session_state["frame_stride"])
    poisson_max_iter = int(st.session_state["poisson_max_iter"])

    if nx < 11 or ny < 11:
        raise ValueError("Grid sizes nx and ny must be at least 11.")
    if reynolds <= 0.0 or lid_velocity <= 0.0 or final_time <= 0.0:
        raise ValueError("Reynolds number, lid velocity, and final time must be positive.")
    if not (0.0 < cfl_number <= 1.0):
        raise ValueError("The CFL number should be in the range (0, 1].")
    if frame_stride < 1 or poisson_max_iter < 1:
        raise ValueError("Frame stride and Poisson iterations must be positive integers.")

    kwargs = {
        "nx": nx,
        "ny": ny,
        "reynolds": reynolds,
        "lid_velocity": lid_velocity,
        "final_time": final_time,
        "cfl_number": cfl_number,
        "frame_stride": frame_stride,
        "poisson_max_iter": poisson_max_iter,
        "poisson_solver": "factorized",
        "animation_interval_ms": 80,
    }

    if mode == "flow":
        return mode, SolverConfig(**kwargs)

    thermal_diffusivity = float(st.session_state["thermal_diffusivity"])
    thermal_conductivity = float(st.session_state["thermal_conductivity"])
    if thermal_diffusivity <= 0.0 or thermal_conductivity <= 0.0:
        raise ValueError("Thermal diffusivity and thermal conductivity must be positive.")

    return mode, ThermalSolverConfig(
        **kwargs,
        thermal_diffusivity=thermal_diffusivity,
        thermal_conductivity=thermal_conductivity,
        initial_temperature=float(st.session_state["initial_temperature"]),
        top_temperature=float(st.session_state["top_temperature"]),
        left_temperature=float(st.session_state["left_temperature"]),
        right_temperature=float(st.session_state["right_temperature"]),
        bottom_bc_mode=str(st.session_state["bottom_bc_mode"]).strip().lower(),
        bottom_temperature=float(st.session_state["bottom_temperature"]),
        bottom_heat_flux=float(st.session_state["bottom_heat_flux"]),
        volumetric_heat_source=float(st.session_state["volumetric_heat_source"]),
        source_x_min=float(st.session_state["source_x_min"]),
        source_x_max=float(st.session_state["source_x_max"]),
        source_y_min=float(st.session_state["source_y_min"]),
        source_y_max=float(st.session_state["source_y_max"]),
    )


def build_record(
    mode: str,
    solver: LidDrivenCavitySolver | LidDrivenCavityThermalSolver,
    snapshots: list[FlowSnapshot | ThermalFlowSnapshot],
) -> SimulationRecord:
    pressure_fields = [frame.pressure for frame in snapshots if frame.pressure is not None]
    omega_samples = np.concatenate([np.abs(frame.omega[1:-1, 1:-1]).ravel() for frame in snapshots])

    ranges = {
        "velocity": (0.0, max(float(max(np.max(frame.velocity_magnitude) for frame in snapshots)), 1.0e-10)),
        "psi": safe_range(
            float(min(np.min(frame.psi) for frame in snapshots)),
            float(max(np.max(frame.psi) for frame in snapshots)),
        ),
        "omega": safe_range(
            -max(float(np.percentile(omega_samples, 99.0)), 1.0e-10),
            max(float(np.percentile(omega_samples, 99.0)), 1.0e-10),
            pad=1.0e-10,
        ),
        "u": safe_range(
            -float(max(np.max(np.abs(frame.u)) for frame in snapshots)),
            float(max(np.max(np.abs(frame.u)) for frame in snapshots)),
        ),
        "v": safe_range(
            -float(max(np.max(np.abs(frame.v)) for frame in snapshots)),
            float(max(np.max(np.abs(frame.v)) for frame in snapshots)),
        ),
        "pressure": safe_range(
            float(min(np.min(field) for field in pressure_fields)),
            float(max(np.max(field) for field in pressure_fields)),
        ),
    }

    levels = {name: np.linspace(vmin, vmax, 32) for name, (vmin, vmax) in ranges.items()}
    history = {
        "time": np.array([frame.time for frame in snapshots], dtype=float),
        "max_velocity": np.array([np.max(frame.velocity_magnitude) for frame in snapshots], dtype=float),
        "max_vorticity": np.array([np.max(np.abs(frame.omega[1:-1, 1:-1])) for frame in snapshots], dtype=float),
    }

    if mode == "thermal":
        temperatures = [frame.temperature for frame in snapshots if isinstance(frame, ThermalFlowSnapshot)]
        ranges["temperature"] = safe_range(
            float(min(np.min(field) for field in temperatures)),
            float(max(np.max(field) for field in temperatures)),
        )
        levels["temperature"] = np.linspace(*ranges["temperature"], 32)
        history["max_temperature"] = np.array([np.max(frame.temperature) for frame in snapshots], dtype=float)

    return SimulationRecord(
        mode=mode,
        config=solver.config,
        x=solver.x.copy(),
        y=solver.y.copy(),
        X=solver.X.copy(),
        Y=solver.Y.copy(),
        snapshots=snapshots,
        ranges=ranges,
        levels=levels,
        history=history,
        signature=signature(mode, solver.config),
    )


def run_simulation(mode: str, config: SolverConfig | ThermalSolverConfig, status_box: Any, summary_box: Any, progress_bar: Any) -> SimulationRecord:
    progress_bar.progress(0)
    status_box.markdown("**Preparing solver...**")
    summary_box.caption("Transient fields will be marched first, then each saved frame will recover pressure.")

    solver: LidDrivenCavitySolver | LidDrivenCavityThermalSolver
    solver = LidDrivenCavitySolver(config) if mode == "flow" else LidDrivenCavityThermalSolver(config)

    def progress_callback(info: dict[str, float]) -> None:
        progress_bar.progress(min(int(84.0 * float(info.get("progress", 0.0))), 84))
        status_box.markdown(
            "**Marching in time**  \n"
            f"step = {int(info.get('step', 0))}  |  time = {float(info.get('time', 0.0)):.3f} s  |  dt = {float(info.get('dt', 0.0)):.3e} s"
        )
        summary = (
            f"max|u| = {float(info.get('max_velocity', 0.0)):.3f} m/s  |  "
            f"max|omega| = {float(info.get('max_vorticity', 0.0)):.3f} s^-1"
        )
        if mode == "thermal":
            summary += f"  |  max T = {float(info.get('max_temperature', 0.0)):.3f} K"
        summary_box.caption(summary)

    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
        snapshots = solver.run(progress_callback=progress_callback)

    total = max(len(snapshots), 1)
    for idx, snapshot in enumerate(snapshots, start=1):
        snapshot.pressure = solver.recover_pressure_field(snapshot)
        progress_bar.progress(min(84 + int(16.0 * idx / total), 100))
        status_box.markdown(f"**Recovering pressure field**  \nframe {idx} / {total}")

    progress_bar.progress(100)
    status_box.markdown("**Computation complete.**")
    summary_box.caption(f"{len(snapshots)} saved frames are ready up to t = {snapshots[-1].time:.3f} s.")
    return build_record(mode, solver, snapshots)


def figure_to_png(figure: Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, facecolor=figure.get_facecolor(), bbox_inches="tight")
    return buffer.getvalue()


def style_axis(axis: Any, config: SolverConfig | ThermalSolverConfig) -> None:
    axis.set_xlim(0.0, config.lx)
    axis.set_ylim(0.0, config.ly)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal")
    axis.tick_params(direction="out", length=4, width=0.8)
    axis.set_facecolor(PALETTE["panel"])


def add_time_banner(figure: Figure, fraction: float, label: str) -> None:
    x0, y0, width = 0.06, 0.905, 0.88
    figure.add_artist(Rectangle((x0, y0), width, 0.008, transform=figure.transFigure, linewidth=0, facecolor=PALETTE["accent_soft"], zorder=20))
    figure.add_artist(Rectangle((x0, y0), width * max(float(np.clip(fraction, 0.0, 1.0)), 0.002), 0.008, transform=figure.transFigure, linewidth=0, facecolor=PALETTE["accent"], zorder=21))
    figure.text(0.94, 0.955, label, ha="right", va="center", fontsize=12.2, fontweight="bold", color=PALETTE["ink"], bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f7f3eb", "edgecolor": "#d6d3d1", "linewidth": 0.9})


def build_overview_figure(record: SimulationRecord, frame_index: int) -> Figure:
    snapshot = record.snapshots[frame_index]
    figure = Figure(figsize=(12.0, 7.4), facecolor=PALETTE["paper"])
    grid = figure.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.30)
    velocity_ax = figure.add_subplot(grid[:, 0])
    psi_ax = figure.add_subplot(grid[0, 1])
    omega_ax = figure.add_subplot(grid[1, 1])

    figure.text(0.06, 0.965, "2-D Lid-Driven Cavity: Flow Overview", ha="left", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    figure.text(0.06, 0.928, "Velocity magnitude and streamlines reveal the main cell, while streamfunction and vorticity expose the rotational structure.", ha="left", fontsize=10.3, color=PALETTE["muted"])
    add_time_banner(figure, snapshot.time / max(record.config.final_time, 1.0e-12), f"t = {snapshot.time:.3f} s")

    velocity_ax.contourf(record.X, record.Y, snapshot.velocity_magnitude, levels=record.levels["velocity"], cmap="jet", extend="max")
    velocity_ax.streamplot(record.x, record.y, snapshot.u, snapshot.v, density=1.0, color="white", linewidth=1.0)
    psi_ax.contourf(record.X, record.Y, snapshot.psi, levels=record.levels["psi"], cmap="jet")
    omega_ax.contourf(record.X, record.Y, np.clip(snapshot.omega, *record.ranges["omega"]), levels=record.levels["omega"], cmap="jet", extend="both")

    for axis in (velocity_ax, psi_ax, omega_ax):
        style_axis(axis, record.config)

    velocity_ax.set_title("Velocity Magnitude and Streamlines", loc="left", fontsize=12, fontweight="bold")
    psi_ax.set_title("Streamfunction", loc="left", fontsize=12, fontweight="bold")
    omega_ax.set_title("Vorticity", loc="left", fontsize=12, fontweight="bold")
    figure.colorbar(cm.ScalarMappable(norm=Normalize(*record.ranges["velocity"]), cmap="jet"), ax=velocity_ax, fraction=0.046, pad=0.018, label="Velocity Magnitude [m/s]")
    figure.colorbar(cm.ScalarMappable(norm=Normalize(*record.ranges["psi"]), cmap="jet"), ax=psi_ax, fraction=0.046, pad=0.025, label="Streamfunction psi")
    figure.colorbar(cm.ScalarMappable(norm=Normalize(*record.ranges["omega"]), cmap="jet"), ax=omega_ax, fraction=0.046, pad=0.025, label="Vorticity omega", extend="both")
    return figure


def build_detail_figure(record: SimulationRecord, frame_index: int) -> Figure:
    snapshot = record.snapshots[frame_index]
    figure = Figure(figsize=(12.0, 6.2), facecolor=PALETTE["paper"])
    grid = figure.add_gridspec(1, 3, wspace=0.26)
    u_ax = figure.add_subplot(grid[0, 0])
    v_ax = figure.add_subplot(grid[0, 1])
    p_ax = figure.add_subplot(grid[0, 2])

    figure.text(0.055, 0.965, "Velocity Components and Pressure", ha="left", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    figure.text(0.055, 0.928, "Component fields make the lid shear layers easier to read, and pressure is recovered after the transient solve finishes.", ha="left", fontsize=10.3, color=PALETTE["muted"])
    add_time_banner(figure, snapshot.time / max(record.config.final_time, 1.0e-12), f"step = {snapshot.step}")

    u_ax.contourf(record.X, record.Y, snapshot.u, levels=record.levels["u"], cmap="jet", extend="both")
    v_ax.contourf(record.X, record.Y, snapshot.v, levels=record.levels["v"], cmap="jet", extend="both")
    p_ax.contourf(record.X, record.Y, snapshot.pressure, levels=record.levels["pressure"], cmap="jet", extend="both")

    for axis in (u_ax, v_ax, p_ax):
        style_axis(axis, record.config)

    u_ax.set_title("u-Velocity", loc="left", fontsize=12, fontweight="bold")
    v_ax.set_title("v-Velocity", loc="left", fontsize=12, fontweight="bold")
    p_ax.set_title("Pressure", loc="left", fontsize=12, fontweight="bold")
    figure.colorbar(cm.ScalarMappable(norm=Normalize(*record.ranges["u"]), cmap="jet"), ax=u_ax, orientation="horizontal", fraction=0.055, pad=0.10, label="u [m/s]")
    figure.colorbar(cm.ScalarMappable(norm=Normalize(*record.ranges["v"]), cmap="jet"), ax=v_ax, orientation="horizontal", fraction=0.055, pad=0.10, label="v [m/s]")
    figure.colorbar(cm.ScalarMappable(norm=Normalize(*record.ranges["pressure"]), cmap="jet"), ax=p_ax, orientation="horizontal", fraction=0.055, pad=0.10, label="Pressure [Pa]")
    return figure


def build_temperature_figure(record: SimulationRecord, frame_index: int) -> Figure:
    snapshot = record.snapshots[frame_index]
    if not isinstance(snapshot, ThermalFlowSnapshot):
        raise ValueError("Temperature view requires a thermal simulation.")

    figure = Figure(figsize=(12.0, 6.2), facecolor=PALETTE["paper"])
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)
    velocity_ax = figure.add_subplot(grid[0, 0])
    temperature_ax = figure.add_subplot(grid[0, 1])

    figure.text(0.06, 0.965, "Velocity and Temperature", ha="left", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    figure.text(0.06, 0.928, "This couples the cavity flow to advection-diffusion in temperature; the dashed box marks the active source region when q''' is nonzero.", ha="left", fontsize=10.3, color=PALETTE["muted"])
    add_time_banner(figure, snapshot.time / max(record.config.final_time, 1.0e-12), f"Tmax = {np.max(snapshot.temperature):.2f} K")

    velocity_ax.contourf(record.X, record.Y, snapshot.velocity_magnitude, levels=record.levels["velocity"], cmap="jet", extend="max")
    velocity_ax.streamplot(record.x, record.y, snapshot.u, snapshot.v, density=1.0, color="white", linewidth=1.0)
    temperature_ax.contourf(record.X, record.Y, snapshot.temperature, levels=record.levels["temperature"], cmap="jet", extend="both")

    if abs(record.config.volumetric_heat_source) > 0.0:
        width = max(record.config.source_x_max - record.config.source_x_min, 0.0)
        height = max(record.config.source_y_max - record.config.source_y_min, 0.0)
        if width > 0.0 and height > 0.0:
            temperature_ax.add_patch(Rectangle((record.config.source_x_min, record.config.source_y_min), width, height, facecolor="none", edgecolor="white", linewidth=1.5, linestyle="--"))

    for axis in (velocity_ax, temperature_ax):
        style_axis(axis, record.config)

    velocity_ax.set_title("Velocity Magnitude and Streamlines", loc="left", fontsize=12, fontweight="bold")
    temperature_ax.set_title("Temperature", loc="left", fontsize=12, fontweight="bold")
    figure.colorbar(cm.ScalarMappable(norm=Normalize(*record.ranges["velocity"]), cmap="jet"), ax=velocity_ax, orientation="horizontal", fraction=0.055, pad=0.10, label="Velocity Magnitude [m/s]")
    figure.colorbar(cm.ScalarMappable(norm=Normalize(*record.ranges["temperature"]), cmap="jet"), ax=temperature_ax, orientation="horizontal", fraction=0.055, pad=0.10, label="Temperature [K]")
    return figure


def build_history_figure(record: SimulationRecord, frame_index: int) -> Figure:
    snapshot = record.snapshots[frame_index]
    series = [
        ("Max speed", record.history["max_velocity"], PALETTE["accent"], "m/s"),
        ("Max |omega|", record.history["max_vorticity"], PALETTE["secondary"], "s^-1"),
    ]
    if record.mode == "thermal":
        series.append(("Max temperature", record.history["max_temperature"], "#2563eb", "K"))

    figure = Figure(figsize=(4.1 * len(series), 3.5), facecolor=PALETTE["paper"])
    grid = figure.add_gridspec(1, len(series), wspace=0.28)
    for idx, (title, values, color, unit) in enumerate(series):
        axis = figure.add_subplot(grid[0, idx])
        axis.plot(record.history["time"], values, color=color, linewidth=2.4)
        axis.axvline(snapshot.time, color=PALETTE["ink"], linestyle="--", linewidth=1.1)
        axis.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
        axis.set_xlabel("time [s]")
        axis.set_ylabel(unit)
        axis.grid(True, linestyle="--", linewidth=0.75, alpha=0.28)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_facecolor(PALETTE["panel"])
    figure.suptitle("Transient history across the saved frames", fontsize=14, fontweight="bold", y=0.99)
    return figure


def render_sidebar() -> tuple[bool, Any, Any, Any]:
    st.sidebar.title("Simulation Controls")
    st.sidebar.caption("Unit-square cavity, streamfunction-vorticity flow solve, optional thermal transport.")
    preset = st.sidebar.selectbox("Teaching case", list(PRESETS.keys()), key="preset")
    if st.session_state["_last_preset"] != preset:
        apply_preset(preset)
        st.session_state["_last_preset"] = preset
    st.sidebar.caption(PRESETS[preset]["description"])
    if st.sidebar.button("Restore selected preset", use_container_width=True):
        apply_preset(preset)
        st.rerun()

    st.sidebar.radio("Physics mode", ["flow", "thermal"], key="mode", horizontal=True, format_func=lambda value: "Flow only" if value == "flow" else "Flow + temperature")
    with st.sidebar.expander("Flow parameters", expanded=True):
        c1, c2 = st.columns(2)
        c1.number_input("nx", min_value=11, step=10, key="nx")
        c2.number_input("ny", min_value=11, step=10, key="ny")
        st.number_input("Reynolds number", min_value=1.0, step=50.0, key="reynolds")
        st.number_input("Lid velocity [m/s]", min_value=0.01, step=0.1, key="lid_velocity")
        st.number_input("Final time [s]", min_value=0.01, step=0.5, key="final_time")
        st.number_input("CFL number", min_value=0.01, max_value=1.0, step=0.05, key="cfl_number")
        st.number_input("Frame stride", min_value=1, step=10, key="frame_stride")
        st.number_input("Poisson iterations", min_value=1, step=10, key="poisson_max_iter")

    if st.session_state["mode"] == "thermal":
        with st.sidebar.expander("Thermal parameters", expanded=True):
            st.number_input("Thermal diffusivity [m^2/s]", min_value=1.0e-8, step=1.0e-4, format="%.6f", key="thermal_diffusivity")
            st.number_input("Thermal conductivity [W/(m K)]", min_value=1.0e-8, step=0.1, key="thermal_conductivity")
            st.number_input("Initial temperature [K]", step=1.0, key="initial_temperature")
            st.number_input("Top wall temperature [K]", step=1.0, key="top_temperature")
            st.number_input("Left wall temperature [K]", step=1.0, key="left_temperature")
            st.number_input("Right wall temperature [K]", step=1.0, key="right_temperature")
            st.selectbox("Bottom boundary mode", ["fixed_temperature", "heat_flux"], key="bottom_bc_mode")
            st.number_input("Bottom temperature [K]", step=1.0, key="bottom_temperature")
            st.number_input("Bottom heat flux [W/m^2]", step=10.0, key="bottom_heat_flux")
        with st.sidebar.expander("Internal heat source", expanded=False):
            st.number_input("Volumetric heat source [W/m^3]", step=1.0e5, format="%.3e", key="volumetric_heat_source")
            c1, c2 = st.columns(2)
            c1.number_input("x min [m]", step=0.05, key="source_x_min")
            c2.number_input("x max [m]", step=0.05, key="source_x_max")
            c1.number_input("y min [m]", step=0.05, key="source_y_min")
            c2.number_input("y max [m]", step=0.05, key="source_y_max")

    compute = st.sidebar.button("Compute Solution", type="primary", use_container_width=True)
    if st.sidebar.button("Clear Results", use_container_width=True):
        st.session_state["result"] = None
        st.session_state["frame_index"] = 0
        st.rerun()
    status_box = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    summary_box = st.sidebar.empty()
    return compute, status_box, summary_box, progress_bar


def render_intro() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f6efe4 0%, #eef3f9 40%, #fbfdff 100%); }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8f4ec 0%, #edf3fb 100%); }
        .hero { background: rgba(255,255,255,0.92); border: 1px solid rgba(19,34,56,0.08); border-radius: 18px; padding: 1.2rem 1.3rem; box-shadow: 0 16px 40px rgba(19,34,56,0.08); margin-bottom: 1rem; }
        .hero h1 { color: #132238; margin: 0 0 0.35rem 0; }
        .hero p { color: #5b6b82; margin: 0; line-height: 1.55; }
        </style>
        <div class="hero">
        <h1>Cavity Flow Teaching Studio</h1>
        <p>This file now uses Streamlit as the interface layer while preserving the existing cavity-flow solver, thermal extension, and teaching-oriented multi-panel visualization workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.info("Overview Fields: velocity magnitude, streamlines, streamfunction, and vorticity.")
    b.info("Velocity and Pressure: component fields plus the post-processed pressure solution.")
    c.info("Temperature Field: the thermal extension with wall heating and interior source support.")


def render_results(record: SimulationRecord) -> None:
    frame_max = len(record.snapshots) - 1
    st.session_state["frame_index"] = int(np.clip(st.session_state["frame_index"], 0, frame_max))
    snapshot = record.snapshots[st.session_state["frame_index"]]

    try:
        current_mode, current_config = build_config()
        if signature(current_mode, current_config) != record.signature:
            st.warning("The sidebar inputs have changed since the last run. The plots below still show the previously computed solution until you recompute.")
    except Exception:
        pass

    cols = st.columns(5 if record.mode == "thermal" else 4)
    cols[0].metric("Time", f"{snapshot.time:.3f} s")
    cols[1].metric("Saved Frame", f"{st.session_state['frame_index'] + 1}/{len(record.snapshots)}")
    cols[2].metric("Max Speed", f"{np.max(snapshot.velocity_magnitude):.3f} m/s")
    cols[3].metric("Max |omega|", f"{np.max(np.abs(snapshot.omega[1:-1, 1:-1])):.3f} s^-1")
    if record.mode == "thermal" and isinstance(snapshot, ThermalFlowSnapshot):
        cols[4].metric("Max Temperature", f"{np.max(snapshot.temperature):.2f} K")

    st.progress(snapshot.time / max(record.config.final_time, 1.0e-12))
    c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 4.0])
    if c1.button("Prev Frame", use_container_width=True, disabled=st.session_state["frame_index"] == 0):
        st.session_state["frame_index"] -= 1
        st.rerun()
    if c2.button("Next Frame", use_container_width=True, disabled=st.session_state["frame_index"] >= frame_max):
        st.session_state["frame_index"] += 1
        st.rerun()
    if c3.button("Reset", use_container_width=True, disabled=st.session_state["frame_index"] == 0):
        st.session_state["frame_index"] = 0
        st.rerun()
    c4.slider("Saved frame", 0, frame_max, key="frame_index")

    views = ["Overview Fields", "Velocity and Pressure"] + (["Temperature Field"] if record.mode == "thermal" else [])
    if st.session_state["view_name"] not in views:
        st.session_state["view_name"] = views[0]
    st.radio("Visualization", views, key="view_name", horizontal=True)
    figure = build_overview_figure(record, st.session_state["frame_index"]) if st.session_state["view_name"] == "Overview Fields" else build_detail_figure(record, st.session_state["frame_index"]) if st.session_state["view_name"] == "Velocity and Pressure" else build_temperature_figure(record, st.session_state["frame_index"])
    st.pyplot(figure, use_container_width=True)
    st.download_button("Download current panel as PNG", data=figure_to_png(figure), file_name=f"cavity_{record.mode}_{st.session_state['view_name'].lower().replace(' ', '_')}_frame_{st.session_state['frame_index'] + 1:03d}.png", mime="image/png")

    with st.expander("Transient history", expanded=True):
        st.pyplot(build_history_figure(record, st.session_state["frame_index"]), use_container_width=True)
    with st.expander("Field guide and governing equations", expanded=False):
        st.markdown("The solver uses the streamfunction-vorticity form of incompressible flow, then reconstructs the velocity field from the streamfunction.")
        st.latex(r"\nabla^2 \psi = -\omega")
        st.latex(r"\frac{\partial \omega}{\partial t} + u \frac{\partial \omega}{\partial x} + v \frac{\partial \omega}{\partial y} = \nu \nabla^2 \omega")
        st.latex(r"u = \frac{\partial \psi}{\partial y}, \qquad v = -\frac{\partial \psi}{\partial x}")
        if record.mode == "thermal":
            st.markdown("The thermal extension adds transient advection-diffusion with an optional volumetric source term.")
            st.latex(r"\frac{\partial T}{\partial t} + u \frac{\partial T}{\partial x} + v \frac{\partial T}{\partial y} = \alpha \nabla^2 T + \frac{q'''}{\rho c_p}")


def main() -> None:
    st.set_page_config(page_title="Cavity Flow Teaching Studio", layout="wide", initial_sidebar_state="expanded")
    init_state()
    render_intro()
    compute, status_box, summary_box, progress_bar = render_sidebar()

    if compute:
        try:
            mode, config = build_config()
            with st.spinner("Running the cavity solver and preparing saved frames..."):
                st.session_state["result"] = run_simulation(mode, config, status_box, summary_box, progress_bar)
            st.session_state["frame_index"] = 0
            st.success("Simulation finished. The saved frames are ready below.")
        except Exception as exc:
            progress_bar.progress(0)
            status_box.markdown("**Computation stopped.**")
            summary_box.caption(str(exc))
            st.error(str(exc))

    record = st.session_state.get("result")
    if record is None:
        status_box.markdown("**Ready.**")
        summary_box.caption("No solution has been computed yet.")
        return

    status_box.markdown("**Last solution loaded.**")
    summary_box.caption(f"{len(record.snapshots)} frames available up to t = {record.snapshots[-1].time:.3f} s.")
    progress_bar.progress(100)
    render_results(record)


if __name__ == "__main__":
    main()
