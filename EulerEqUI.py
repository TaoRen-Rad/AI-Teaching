import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from dataclasses import dataclass


PALETTE = {
    "paper": "#f8f6f2",
    "panel": "#ffffff",
    "sidebar": "#f1ebdf",
    "ink": "#1f2937",
    "muted": "#6b7280",
    "grid": "#d8dee6",
    "accent": "#0f766e",
}

PRESETS = {
    "Triangular pulse (analytic reference)": {
        "expression": "np.where((x >= -1.0) & (x < 0.0), 1.0 + x, np.where((x >= 0.0) & (x <= 1.0), 1.0 - x, 0.0))",
        "description": "Classical compact triangular pulse. Uses the analytic reference solution when the expression is unchanged.",
        "reference_mode": "analytic_triangle",
    },
    "Square pulse": {
        "expression": "np.where((x >= -1.0) & (x <= 1.0), 1.0, 0.0)",
        "description": "A discontinuous top-hat profile, useful for observing shock formation and numerical diffusion.",
        "reference_mode": "fine_grid",
    },
    "Gaussian bump": {
        "expression": "np.exp(-x**2)",
        "description": "A smooth bump that steepens over time and later develops a shock-like front.",
        "reference_mode": "fine_grid",
    },
    "Cosine hat": {
        "expression": "np.where(np.abs(x) <= 2.0, 0.5 * (1.0 + np.cos(np.pi * x / 2.0)), 0.0)",
        "description": "A smooth compactly supported initial profile with gradual steepening.",
        "reference_mode": "fine_grid",
    },
    "Double bump": {
        "expression": "0.8*np.exp(-(x+1.3)**2/0.25) + 0.55*np.exp(-(x-0.9)**2/0.16)",
        "description": "Two interacting positive waves. This is useful for testing more complex nonlinear interactions.",
        "reference_mode": "fine_grid",
    },
    "Custom expression": {
        "expression": "np.where((x >= 0.0), 1.0, 0.0)",
        "description": "Edit the expression freely using x and NumPy functions such as np.where, np.sin, np.exp, np.maximum, and np.clip.",
        "reference_mode": "fine_grid",
    },
}

ALLOWED_SYMBOLS = {
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "tanh": np.tanh,
    "where": np.where,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "clip": np.clip,
    "pi": np.pi,
}


@dataclass
class BurgersConfig:
    xmin: float = -4.0
    xmax: float = 10.0
    nx: int = 501
    t_end: float = 5.0
    frames: int = 121
    cfl: float = 0.72


def evaluate_initial_condition(expression, x):
    local_env = dict(ALLOWED_SYMBOLS)
    local_env["x"] = x
    values = eval(expression, {"__builtins__": {}}, local_env)
    array = np.asarray(values, dtype=float)
    if array.shape == ():
        array = np.full_like(x, float(array), dtype=float)
    if array.shape != x.shape:
        raise ValueError("The initial condition expression must evaluate to an array with the same shape as x.")
    return array


def burgers_exact_triangle(x, time):
    x = np.asarray(x)
    u = np.zeros_like(x, dtype=float)
    if time <= 1.0:
        mask1 = (x >= -1.0) & (x < time)
        mask2 = (x >= time) & (x <= 1.0)
        u[mask1] = (1.0 + x[mask1]) / (1.0 + time)
        u[mask2] = (1.0 - x[mask2]) / (1.0 - time)
    else:
        x_s = np.sqrt(2.0 * (1.0 + time)) - 1.0
        mask = (x >= -1.0) & (x < x_s)
        u[mask] = (1.0 + x[mask]) / (1.0 + time)
    return u


def step_conservative(u, dt, dx, left_bc, right_bc):
    sigma = dt / dx
    un = u.copy()
    um = np.roll(un, 1)
    up = np.roll(un, -1)
    am = 0.5 * (un + um)
    ap = 0.5 * (up + un)
    u_new = (
        un
        - 0.5 * sigma * (am + np.abs(am)) * (un - um)
        - 0.5 * sigma * (ap - np.abs(ap)) * (up - un)
    )
    u_new[0] = left_bc
    u_new[-1] = right_bc
    return u_new


def step_nonconservative(u, dt, dx, left_bc, right_bc):
    sigma = dt / dx
    un = u.copy()
    um = np.roll(un, 1)
    up = np.roll(un, -1)
    du = np.where(un >= 0.0, un - um, up - un)
    u_new = un - sigma * un * du
    u_new[0] = left_bc
    u_new[-1] = right_bc
    return u_new


def solve_scheme_snapshots(x, output_times, u0, scheme_name, cfl):
    dx = x[1] - x[0]
    left_bc = u0[0]
    right_bc = u0[-1]
    current = u0.copy()
    snapshots = np.zeros((len(output_times), len(x)), dtype=float)
    snapshots[0] = current
    t_now = 0.0
    frame = 0
    step_fn = step_conservative if scheme_name == "conservative" else step_nonconservative
    while frame < len(output_times) - 1:
        t_target = output_times[frame + 1]
        while t_now < t_target - 1e-12:
            speed = max(np.max(np.abs(current)), 1e-8)
            dt = min(cfl * dx / speed, t_target - t_now)
            current = step_fn(current, dt, dx, left_bc, right_bc)
            t_now += dt
        frame += 1
        snapshots[frame] = current
    return snapshots


def compute_reference_snapshots(config, output_times, expression, preset_name, current_expression):
    x = np.linspace(config.xmin, config.xmax, config.nx)
    preset = PRESETS[preset_name]
    if preset["reference_mode"] == "analytic_triangle" and current_expression.strip() == preset["expression"].strip():
        reference = np.zeros((len(output_times), config.nx), dtype=float)
        for idx, time_value in enumerate(output_times):
            reference[idx] = burgers_exact_triangle(x, time_value)
        return reference, "Analytic exact reference"

    refine_factor = 4
    nx_ref = refine_factor * (config.nx - 1) + 1
    x_ref = np.linspace(config.xmin, config.xmax, nx_ref)
    u0_ref = evaluate_initial_condition(expression, x_ref)
    ref_snapshots_fine = solve_scheme_snapshots(
        x_ref,
        output_times,
        u0_ref,
        scheme_name="conservative",
        cfl=min(0.55, 0.85 * config.cfl),
    )
    reference = np.zeros((len(output_times), config.nx), dtype=float)
    for idx in range(len(output_times)):
        reference[idx] = np.interp(x, x_ref, ref_snapshots_fine[idx])
    return reference, "Fine-grid conservative reference"


def build_results(config, expression, preset_name):
    x = np.linspace(config.xmin, config.xmax, config.nx)
    output_times = np.linspace(0.0, config.t_end, config.frames)
    u0 = evaluate_initial_condition(expression, x)
    reference, reference_label = compute_reference_snapshots(config, output_times, expression, preset_name, expression)
    nonconservative = solve_scheme_snapshots(x, output_times, u0, scheme_name="nonconservative", cfl=config.cfl)
    conservative = solve_scheme_snapshots(x, output_times, u0, scheme_name="conservative", cfl=config.cfl)

    l1_noncons = float(np.mean(np.abs(nonconservative[-1] - reference[-1])))
    l1_cons = float(np.mean(np.abs(conservative[-1] - reference[-1])))
    linf_noncons = float(np.max(np.abs(nonconservative[-1] - reference[-1])))
    linf_cons = float(np.max(np.abs(conservative[-1] - reference[-1])))

    all_values = np.concatenate([reference.ravel(), nonconservative.ravel(), conservative.ravel()])
    plot_min = min(-0.02, float(np.min(all_values)))
    plot_max = max(1.02, float(np.max(all_values)))
    span = max(plot_max - plot_min, 1e-8)

    return {
        "x": x,
        "times": output_times,
        "u0": u0,
        "reference": reference,
        "nonconservative": nonconservative,
        "conservative": conservative,
        "reference_label": reference_label,
        "line_ylim": (plot_min - 0.04 * span, plot_max + 0.06 * span),
        "color_limits": (plot_min, plot_max),
        "metrics": {
            "noncons_l1": l1_noncons,
            "cons_l1": l1_cons,
            "noncons_linf": linf_noncons,
            "cons_linf": linf_cons,
        },
    }


@st.cache_data(show_spinner=False)
def cached_build_results(xmin, xmax, nx, t_end, frames, cfl, expression, preset_name):
    config = BurgersConfig(xmin=xmin, xmax=xmax, nx=nx, t_end=t_end, frames=frames, cfl=cfl)
    return build_results(config, expression, preset_name)


def make_figure(results, frame_index):
    cmap = colormaps["jet"].copy()
    cmap.set_bad(color="white")

    x = results["x"]
    times = results["times"]
    ymin, ymax = results["line_ylim"]
    vmin, vmax = results["color_limits"]
    t_value = float(times[frame_index])

    fig = plt.figure(figsize=(11.2, 7.8), facecolor=PALETTE["paper"])
    grid = fig.add_gridspec(
        2, 3,
        left=0.06, right=0.95, bottom=0.10, top=0.80,
        width_ratios=[1.0, 1.0, 0.05],
        height_ratios=[0.92, 1.12],
        hspace=0.24, wspace=0.14,
    )
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])
    cax = fig.add_subplot(grid[:, 2])

    fig.text(0.06, 0.965, "Burgers Equation: Conservative vs Non-conservative Forms",
             ha="left", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.06, 0.925,
             "Top row: snapshots against the reference solution. Bottom row: space-time evolution with one shared color scale.",
             ha="left", fontsize=10.4, color=PALETTE["muted"])
    fig.text(0.94, 0.955, f"t = {t_value:.2f}", ha="right", va="center", fontsize=12.5,
             fontweight="bold", color=PALETTE["ink"],
             bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f7f3eb", "edgecolor": "#d6d3d1", "linewidth": 0.9})

    progress = 1.0 if times[-1] <= 0 else t_value / times[-1]
    fig.add_artist(Rectangle((0.06, 0.895), 0.89, 0.007, transform=fig.transFigure,
                             linewidth=0, facecolor="#d8e5e2", zorder=20))
    fig.add_artist(Rectangle((0.06, 0.895), 0.89 * max(progress, 0.002), 0.007, transform=fig.transFigure,
                             linewidth=0, facecolor=PALETTE["accent"], zorder=21))

    legend_lines = [
        Line2D([0], [0], color="#111827", lw=2.6),
        Line2D([0], [0], color="#2563eb", lw=2.3, ls="--"),
        Line2D([0], [0], color="#d9480f", lw=2.3, ls="--"),
    ]
    fig.legend(legend_lines, ["Reference", "Non-conservative", "Conservative"],
               loc="upper center", bbox_to_anchor=(0.44, 0.895), ncol=3, frameon=False,
               handlelength=2.0, columnspacing=2.0)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_facecolor(PALETTE["panel"])
        ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.set_title("Non-conservative form", loc="left", fontweight="bold")
    ax2.set_title("Conservative form", loc="left", fontweight="bold")
    ax3.set_title("Space-time evolution: Non-conservative", loc="left", fontweight="bold")
    ax4.set_title("Space-time evolution: Conservative", loc="left", fontweight="bold")

    ax1.plot(x, results["reference"][frame_index], color="#111827", lw=2.6)
    ax1.plot(x, results["nonconservative"][frame_index], color="#2563eb", lw=2.3, ls="--")
    ax2.plot(x, results["reference"][frame_index], color="#111827", lw=2.6)
    ax2.plot(x, results["conservative"][frame_index], color="#d9480f", lw=2.3, ls="--")

    for ax in (ax1, ax2):
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")

    mask_noncons = np.ones_like(results["nonconservative"], dtype=bool)
    mask_noncons[:frame_index + 1, :] = False
    mask_cons = np.ones_like(results["conservative"], dtype=bool)
    mask_cons[:frame_index + 1, :] = False

    img_noncons = ax3.imshow(np.ma.array(results["nonconservative"], mask=mask_noncons), origin="lower",
                             extent=[float(x[0]), float(x[-1]), float(times[0]), float(times[-1])],
                             aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax4.imshow(np.ma.array(results["conservative"], mask=mask_cons), origin="lower",
               extent=[float(x[0]), float(x[-1]), float(times[0]), float(times[-1])],
               aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    for ax in (ax3, ax4):
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(float(times[0]), float(times[-1]))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        ax.axhline(t_value, color=PALETTE["accent"], lw=1.5, ls="--")

    cbar = fig.colorbar(img_noncons, cax=cax)
    cbar.set_label("Solution value $u$")
    return fig


def main():
    st.set_page_config(page_title="Burgers Equation Studio", layout="wide")
    st.title("Burgers Equation Studio")
    st.caption("Streamlit version migrated from a Tkinter teaching demo.")

    with st.sidebar:
        st.header("Initial condition")
        preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
        st.write(PRESETS[preset_name]["description"])
        expression = st.text_area("u0(x) expression", value=PRESETS[preset_name]["expression"], height=120)

        st.header("Numerical setup")
        xmin = st.number_input("x_min", value=-4.0)
        xmax = st.number_input("x_max", value=10.0)
        nx = st.number_input("Nx", min_value=101, max_value=2001, value=501, step=50)
        t_end = st.number_input("t_end", min_value=0.1, value=5.0, step=0.1)
        frames = st.number_input("frames", min_value=20, max_value=400, value=121, step=10)
        cfl = st.number_input("CFL", min_value=0.05, max_value=1.5, value=0.72, step=0.01, format="%.2f")
        run = st.button("Compute", use_container_width=True)

    if "results" not in st.session_state:
        st.info("Pick a preset or edit the initial condition, then click **Compute**.")

    if run:
        try:
            x_preview = np.linspace(float(xmin), float(xmax), int(nx))
            evaluate_initial_condition(expression, x_preview)
            with st.spinner("Computing reference, non-conservative, and conservative solutions..."):
                st.session_state.results = cached_build_results(
                    float(xmin), float(xmax), int(nx), float(t_end), int(frames), float(cfl), expression, preset_name
                )
        except Exception as exc:
            st.error(f"Computation failed: {exc}")

    results = st.session_state.get("results")
    if results is None:
        return

    col1, col2 = st.columns([1.15, 0.85])
    with col2:
        st.subheader("Reference and errors")
        st.write(f"**Reference mode:** {results['reference_label']}")
        st.metric("Non-conservative L1", f"{results['metrics']['noncons_l1']:.4e}")
        st.metric("Conservative L1", f"{results['metrics']['cons_l1']:.4e}")
        st.write(
            f"Non-conservative Linf = {results['metrics']['noncons_linf']:.4e}  \\\nConservative Linf = {results['metrics']['cons_linf']:.4e}"
        )
        frame_index = st.slider("Frame", min_value=0, max_value=len(results["times"]) - 1, value=0)
        st.write(f"Time = {results['times'][frame_index]:.2f} / {results['times'][-1]:.2f}")

    with col1:
        fig = make_figure(results, frame_index)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander("Teaching note"):
        st.write(
            "This app compares non-conservative and conservative updates for the Burgers equation. "
            "The top row shows the current snapshot against a reference solution, while the bottom row shows the space-time history up to the selected frame."
        )


if __name__ == "__main__":
    main()
