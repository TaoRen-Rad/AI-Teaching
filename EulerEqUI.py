import argparse
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


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
    if (
        preset["reference_mode"] == "analytic_triangle"
        and current_expression.strip() == preset["expression"].strip()
    ):
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


def build_results(config, expression, preset_name, progress_callback=None):
    x = np.linspace(config.xmin, config.xmax, config.nx)
    output_times = np.linspace(0.0, config.t_end, config.frames)
    u0 = evaluate_initial_condition(expression, x)

    if progress_callback is not None:
        progress_callback("Computing reference solution...", 0.15)
    reference, reference_label = compute_reference_snapshots(
        config,
        output_times,
        expression,
        preset_name,
        expression,
    )

    if progress_callback is not None:
        progress_callback("Computing non-conservative form...", 0.45)
    nonconservative = solve_scheme_snapshots(
        x, output_times, u0, scheme_name="nonconservative", cfl=config.cfl
    )

    if progress_callback is not None:
        progress_callback("Computing conservative form...", 0.75)
    conservative = solve_scheme_snapshots(
        x, output_times, u0, scheme_name="conservative", cfl=config.cfl
    )

    if progress_callback is not None:
        progress_callback("Preparing visualization...", 1.0)

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


class EulerEqUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Burgers Equation Studio")
        self.root.geometry("1600x960")
        self.root.minsize(1420, 860)
        self.root.configure(bg=PALETTE["paper"])

        self.results = None
        self.current_frame = 0
        self.playing = False
        self.after_id = None
        self.shared_cmap = colormaps["jet"].copy()
        self.shared_cmap.set_bad(color="white")

        self.preset_var = tk.StringVar(value="Triangular pulse (analytic reference)")
        self.xmin_var = tk.DoubleVar(value=-4.0)
        self.xmax_var = tk.DoubleVar(value=10.0)
        self.nx_var = tk.IntVar(value=501)
        self.tend_var = tk.DoubleVar(value=5.0)
        self.frames_var = tk.IntVar(value=121)
        self.cfl_var = tk.DoubleVar(value=0.72)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.slider_var = tk.DoubleVar(value=0.0)

        self.status_var = tk.StringVar(value="Load an example or edit the expression, then click Compute.")
        self.reference_var = tk.StringVar(value="Reference: analytic exact for the default triangular example.")
        self.metrics_var = tk.StringVar(value="Final errors will appear here after computation.")
        self.time_var = tk.StringVar(value="t = 0.00 / 0.00")
        self.description_var = tk.StringVar(value=PRESETS[self.preset_var.get()]["description"])

        self._configure_style()
        self._build_layout()
        self._build_figure()
        self.load_preset()

    def _configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Main.TFrame", background=PALETTE["paper"])
        style.configure("Sidebar.TFrame", background=PALETTE["sidebar"])
        style.configure("Sidebar.TLabel", background=PALETTE["sidebar"], foreground=PALETTE["ink"])
        style.configure("Muted.TLabel", background=PALETTE["sidebar"], foreground=PALETTE["muted"])
        style.configure("Accent.TButton", background=PALETTE["accent"], foreground="white", padding=(10, 6), borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#0b5c56"), ("disabled", "#7aa8a2")])
        style.configure("Soft.TButton", background="#e8edf2", foreground=PALETTE["ink"], padding=(10, 6))
        style.configure("Panel.TLabelframe", background=PALETTE["sidebar"], foreground=PALETTE["ink"])
        style.configure("Panel.TLabelframe.Label", background=PALETTE["sidebar"], foreground=PALETTE["ink"], font=("Segoe UI", 10, "bold"))
        style.configure(
            "Compute.Horizontal.TProgressbar",
            troughcolor="#d9e5e5",
            background=PALETTE["accent"],
            bordercolor="#d9e5e5",
            lightcolor=PALETTE["accent"],
            darkcolor=PALETTE["accent"],
        )

    def _build_layout(self):
        main = ttk.Frame(self.root, style="Main.TFrame")
        main.pack(fill="both", expand=True)

        self.sidebar_container = ttk.Frame(main, style="Sidebar.TFrame", width=370)
        self.sidebar_container.pack(side="left", fill="y")
        self.sidebar_container.pack_propagate(False)

        self.sidebar_canvas = tk.Canvas(self.sidebar_container, bg=PALETTE["sidebar"], highlightthickness=0, bd=0)
        self.sidebar_scroll = ttk.Scrollbar(self.sidebar_container, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scroll.set)
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        self.sidebar_scroll.pack(side="right", fill="y")

        self.sidebar = ttk.Frame(self.sidebar_canvas, style="Sidebar.TFrame")
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
        self.sidebar.bind("<Configure>", self._on_sidebar_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)
        self.sidebar_canvas.bind("<Enter>", self._bind_mousewheel)
        self.sidebar_canvas.bind("<Leave>", self._unbind_mousewheel)

        self.content = ttk.Frame(main, style="Main.TFrame")
        self.content.pack(side="right", fill="both", expand=True, padx=(14, 14), pady=(12, 12))

        title = ttk.Label(
            self.sidebar,
            text="Burgers Equation\nUI Studio",
            style="Sidebar.TLabel",
            font=("Segoe UI", 18, "bold"),
            justify="left",
        )
        title.pack(anchor="w", padx=16, pady=(18, 8))

        desc = ttk.Label(
            self.sidebar,
            text="Pick an example, edit the initial condition expression if you want, compute first, then play the animation.",
            style="Muted.TLabel",
            wraplength=312,
            justify="left",
        )
        desc.pack(anchor="w", padx=16, pady=(0, 12))

        self._build_preset_panel()
        self._build_config_panel()
        self._build_action_panel()
        self._build_info_panel()

        footer = ttk.Label(self.sidebar, textvariable=self.status_var, style="Muted.TLabel", wraplength=312, justify="left")
        footer.pack(fill="x", padx=16, pady=(10, 14))

        self.figure_frame = ttk.Frame(self.content, style="Main.TFrame")
        self.figure_frame.pack(fill="both", expand=True)

        playback = ttk.Frame(self.content, style="Main.TFrame")
        playback.pack(fill="x", pady=(8, 0))

        self.play_button = ttk.Button(playback, text="Play", style="Soft.TButton", command=self.toggle_playback)
        self.play_button.pack(side="left")
        self.reset_button = ttk.Button(playback, text="Reset", style="Soft.TButton", command=self.reset_playback)
        self.reset_button.pack(side="left", padx=(8, 10))
        ttk.Label(playback, text="Playback speed", style="Sidebar.TLabel").pack(side="left", padx=(4, 8))
        ttk.Scale(playback, from_=0.4, to=2.5, variable=self.speed_var, orient="horizontal").pack(side="left", ipadx=52)
        ttk.Label(playback, textvariable=self.time_var, style="Sidebar.TLabel").pack(side="right")

        self.frame_scale = ttk.Scale(
            self.content,
            from_=0,
            to=1,
            orient="horizontal",
            variable=self.slider_var,
            command=self._on_slider_changed,
        )
        self.frame_scale.pack(fill="x", pady=(10, 0))

    def _build_preset_panel(self):
        panel = ttk.LabelFrame(self.sidebar, text="Initial Condition", style="Panel.TLabelframe")
        panel.pack(fill="x", padx=16, pady=(0, 10))

        ttk.Label(panel, text="Preset", style="Sidebar.TLabel").pack(anchor="w", padx=10, pady=(10, 4))
        preset_box = ttk.Combobox(panel, textvariable=self.preset_var, values=list(PRESETS.keys()), state="readonly")
        preset_box.pack(fill="x", padx=10)
        preset_box.bind("<<ComboboxSelected>>", lambda _event: self.load_preset())

        ttk.Label(panel, text="Description", style="Sidebar.TLabel").pack(anchor="w", padx=10, pady=(10, 4))
        ttk.Label(panel, textvariable=self.description_var, style="Muted.TLabel", wraplength=300, justify="left").pack(fill="x", padx=10)

        ttk.Label(panel, text="u0(x) expression", style="Sidebar.TLabel").pack(anchor="w", padx=10, pady=(10, 4))
        self.expression_text = tk.Text(panel, height=5, wrap="word", font=("Consolas", 10))
        self.expression_text.pack(fill="x", padx=10, pady=(0, 10))

    def _build_config_panel(self):
        panel = ttk.LabelFrame(self.sidebar, text="Numerical Setup", style="Panel.TLabelframe")
        panel.pack(fill="x", padx=16, pady=(0, 10))

        entries = [
            ("x_min", self.xmin_var),
            ("x_max", self.xmax_var),
            ("Nx", self.nx_var),
            ("t_end", self.tend_var),
            ("frames", self.frames_var),
            ("CFL", self.cfl_var),
        ]
        for idx, (label, var) in enumerate(entries):
            ttk.Label(panel, text=label, style="Sidebar.TLabel").grid(row=idx, column=0, sticky="w", padx=10, pady=4)
            ttk.Entry(panel, textvariable=var, width=12).grid(row=idx, column=1, sticky="ew", padx=10, pady=4)
        panel.columnconfigure(1, weight=1)

    def _build_action_panel(self):
        panel = ttk.LabelFrame(self.sidebar, text="Actions", style="Panel.TLabelframe")
        panel.pack(fill="x", padx=16, pady=(0, 10))

        self.compute_button = ttk.Button(panel, text="Compute", style="Accent.TButton", command=self.compute_all)
        self.compute_button.pack(fill="x", padx=10, pady=(10, 8))

        self.save_button = ttk.Button(
            panel,
            text="Save Current Image",
            style="Soft.TButton",
            command=self.save_current_image,
            state="disabled",
        )
        self.save_button.pack(fill="x", padx=10, pady=(0, 8))

        self.compute_progress = ttk.Progressbar(
            panel,
            orient="horizontal",
            mode="determinate",
            variable=self.progress_var,
            maximum=100.0,
            style="Compute.Horizontal.TProgressbar",
        )
        self.compute_progress.pack(fill="x", padx=10, pady=(0, 10))

    def _build_info_panel(self):
        panel = ttk.LabelFrame(self.sidebar, text="Reference and Errors", style="Panel.TLabelframe")
        panel.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        ttk.Label(panel, textvariable=self.reference_var, style="Muted.TLabel", wraplength=300, justify="left").pack(fill="x", padx=10, pady=(10, 6))
        ttk.Label(panel, textvariable=self.metrics_var, style="Muted.TLabel", wraplength=300, justify="left").pack(fill="x", padx=10, pady=(0, 10))

    def _build_figure(self):
        self.figure = Figure(figsize=(11.2, 7.8), facecolor=PALETTE["paper"])
        grid = self.figure.add_gridspec(
            2,
            3,
            left=0.06,
            right=0.95,
            bottom=0.10,
            top=0.80,
            width_ratios=[1.0, 1.0, 0.05],
            height_ratios=[0.92, 1.12],
            hspace=0.24,
            wspace=0.14,
        )

        self.ax1 = self.figure.add_subplot(grid[0, 0])
        self.ax2 = self.figure.add_subplot(grid[0, 1])
        self.ax3 = self.figure.add_subplot(grid[1, 0])
        self.ax4 = self.figure.add_subplot(grid[1, 1])
        self.cax = self.figure.add_subplot(grid[:, 2])

        self.figure.text(
            0.06,
            0.965,
            "Burgers Equation: Conservative vs Non-conservative Forms",
            ha="left",
            fontsize=18,
            fontweight="bold",
            color=PALETTE["ink"],
        )
        self.figure.text(
            0.06,
            0.925,
            "Top row: snapshots against the reference solution. Bottom row: space-time evolution with one shared color scale.",
            ha="left",
            fontsize=10.4,
            color=PALETTE["muted"],
        )
        self.time_badge = self.figure.text(
            0.94,
            0.955,
            "t = 0.00",
            ha="right",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color=PALETTE["ink"],
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f7f3eb", "edgecolor": "#d6d3d1", "linewidth": 0.9},
        )

        self.progress_bg = Rectangle((0.06, 0.895), 0.89, 0.007, transform=self.figure.transFigure, linewidth=0, facecolor="#d8e5e2", zorder=20)
        self.progress_bar = Rectangle((0.06, 0.895), 0.002, 0.007, transform=self.figure.transFigure, linewidth=0, facecolor=PALETTE["accent"], zorder=21)
        self.figure.add_artist(self.progress_bg)
        self.figure.add_artist(self.progress_bar)

        legend_lines = [
            Line2D([0], [0], color="#111827", lw=2.6),
            Line2D([0], [0], color="#2563eb", lw=2.3, ls="--"),
            Line2D([0], [0], color="#d9480f", lw=2.3, ls="--"),
        ]
        self.legend = self.figure.legend(
            legend_lines,
            ["Reference", "Non-conservative", "Conservative"],
            loc="upper center",
            bbox_to_anchor=(0.44, 0.895),
            ncol=3,
            frameon=False,
            handlelength=2.0,
            columnspacing=2.0,
        )

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.set_facecolor(PALETTE["panel"])
            ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        self.ax1.set_title("Non-conservative form", loc="left", fontweight="bold")
        self.ax2.set_title("Conservative form", loc="left", fontweight="bold")
        self.ax3.set_title("Space-time evolution: Non-conservative", loc="left", fontweight="bold")
        self.ax4.set_title("Space-time evolution: Conservative", loc="left", fontweight="bold")

        x_placeholder = np.linspace(-4.0, 10.0, 50)
        self.line_ref_noncons, = self.ax1.plot(x_placeholder, np.zeros_like(x_placeholder), color="#111827", lw=2.6)
        self.line_noncons, = self.ax1.plot(x_placeholder, np.zeros_like(x_placeholder), color="#2563eb", lw=2.3, ls="--")
        self.line_ref_cons, = self.ax2.plot(x_placeholder, np.zeros_like(x_placeholder), color="#111827", lw=2.6)
        self.line_cons, = self.ax2.plot(x_placeholder, np.zeros_like(x_placeholder), color="#d9480f", lw=2.3, ls="--")

        masked_placeholder = np.ma.masked_all((60, 50))
        self.img_noncons = self.ax3.imshow(masked_placeholder, origin="lower", extent=[-4.0, 10.0, 0.0, 5.0], aspect="auto", cmap=self.shared_cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        self.img_cons = self.ax4.imshow(masked_placeholder, origin="lower", extent=[-4.0, 10.0, 0.0, 5.0], aspect="auto", cmap=self.shared_cmap, vmin=0.0, vmax=1.0, interpolation="nearest")

        self.time_line_noncons = self.ax3.axhline(0.0, color=PALETTE["accent"], lw=1.5, ls="--")
        self.time_line_cons = self.ax4.axhline(0.0, color=PALETTE["accent"], lw=1.5, ls="--")

        self.cbar = self.figure.colorbar(self.img_cons, cax=self.cax)
        self.cbar.set_label("Solution value $u$")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_sidebar_configure(self, _event):
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_sidebar_canvas_configure(self, event):
        self.sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width)

    def _bind_mousewheel(self, _event):
        self.sidebar_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event):
        self.sidebar_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.sidebar_canvas.yview_scroll(int(-event.delta / 120), "units")

    def load_preset(self):
        preset = PRESETS[self.preset_var.get()]
        self.description_var.set(preset["description"])
        self.expression_text.delete("1.0", "end")
        self.expression_text.insert("1.0", preset["expression"])
        if preset["reference_mode"] == "analytic_triangle":
            self.reference_var.set("Reference: analytic exact solution for the standard triangular pulse.")
        else:
            self.reference_var.set("Reference: fine-grid conservative solution, automatically used for general initial conditions.")

    def _read_config(self):
        config = BurgersConfig(
            xmin=float(self.xmin_var.get()),
            xmax=float(self.xmax_var.get()),
            nx=int(self.nx_var.get()),
            t_end=float(self.tend_var.get()),
            frames=int(self.frames_var.get()),
            cfl=float(self.cfl_var.get()),
        )
        if config.xmax <= config.xmin:
            raise ValueError("x_max must be larger than x_min.")
        if config.nx < 101:
            raise ValueError("Please use at least 101 grid points.")
        if config.frames < 20:
            raise ValueError("Please use at least 20 animation frames.")
        if config.t_end <= 0.0:
            raise ValueError("t_end must be positive.")
        if config.cfl <= 0.0:
            raise ValueError("CFL must be positive.")
        return config

    def _read_expression(self):
        expression = self.expression_text.get("1.0", "end").strip()
        if not expression:
            raise ValueError("Please provide a valid initial condition expression.")
        return expression

    def _set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def _build_snapshot_filename(self):
        preset_name = self.preset_var.get().strip().lower()
        preset_tag = "".join(char if char.isalnum() else "_" for char in preset_name).strip("_")
        if not preset_tag:
            preset_tag = "custom"

        if self.results is None:
            return f"burgers_{preset_tag}_snapshot.png"

        time_value = float(self.results["times"][self.current_frame])
        time_tag = f"{time_value:05.2f}".replace(".", "p").replace("-", "m")
        return f"burgers_{preset_tag}_t{time_tag}.png"

    def save_current_image(self):
        if self.results is None:
            messagebox.showinfo("No Data Yet", "Please compute the solution first.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save Current Figure",
            initialfile=self._build_snapshot_filename(),
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF file", "*.pdf"),
                ("SVG file", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not output_path:
            return

        try:
            self.canvas.draw()
            self.figure.savefig(
                output_path,
                dpi=180,
                facecolor=self.figure.get_facecolor(),
                bbox_inches="tight",
            )
            self._set_status(f"Saved current figure to: {output_path}")
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def compute_all(self):
        try:
            config = self._read_config()
            expression = self._read_expression()
            x_preview = np.linspace(config.xmin, config.xmax, config.nx)
            _ = evaluate_initial_condition(expression, x_preview)
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self.stop_playback()
        had_results = self.results is not None
        self.compute_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.progress_var.set(0.0)
        self._set_status("Starting computation...")

        def progress_callback(message, fraction):
            self.progress_var.set(100.0 * fraction)
            self._set_status(message)

        try:
            self.results = build_results(
                config=config,
                expression=expression,
                preset_name=self.preset_var.get(),
                progress_callback=progress_callback,
            )
        except Exception as exc:
            self.compute_button.configure(state="normal")
            self.save_button.configure(state="normal" if had_results else "disabled")
            messagebox.showerror("Computation Failed", str(exc))
            return

        self.metrics_var.set(
            "Final-time errors against the reference\n"
            f"Non-conservative:  L1 = {self.results['metrics']['noncons_l1']:.4e},  Linf = {self.results['metrics']['noncons_linf']:.4e}\n"
            f"Conservative:      L1 = {self.results['metrics']['cons_l1']:.4e},  Linf = {self.results['metrics']['cons_linf']:.4e}"
        )
        self.reference_var.set(f"Reference mode: {self.results['reference_label']}")

        self.frame_scale.configure(to=max(len(self.results["times"]) - 1, 1))
        self.slider_var.set(0.0)
        self.current_frame = 0
        self._configure_plot_ranges()
        self.update_frame(0)

        self.compute_button.configure(state="normal")
        self.save_button.configure(state="normal")
        self.play_button.configure(text="Play")
        self._set_status("Computation finished. Drag the slider or press Play.")

    def _configure_plot_ranges(self):
        x = self.results["x"]
        times = self.results["times"]
        ymin, ymax = self.results["line_ylim"]
        vmin, vmax = self.results["color_limits"]

        for ax in (self.ax1, self.ax2):
            ax.set_xlim(float(x[0]), float(x[-1]))
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$u$")

        for ax in (self.ax3, self.ax4):
            ax.set_xlim(float(x[0]), float(x[-1]))
            ax.set_ylim(float(times[0]), float(times[-1]))
            ax.set_xlabel("$x$")
            ax.set_ylabel("$t$")

        self.img_noncons.set_extent([float(x[0]), float(x[-1]), float(times[0]), float(times[-1])])
        self.img_cons.set_extent([float(x[0]), float(x[-1]), float(times[0]), float(times[-1])])
        self.img_noncons.set_clim(vmin, vmax)
        self.img_cons.set_clim(vmin, vmax)
        self.cbar.update_normal(self.img_cons)

    def update_frame(self, frame_index):
        if self.results is None:
            return

        frame_index = int(np.clip(frame_index, 0, len(self.results["times"]) - 1))
        self.current_frame = frame_index
        x = self.results["x"]
        t_value = float(self.results["times"][frame_index])
        t_end = float(self.results["times"][-1])
        progress = 1.0 if t_end <= 0.0 else t_value / t_end

        self.line_ref_noncons.set_data(x, self.results["reference"][frame_index])
        self.line_noncons.set_data(x, self.results["nonconservative"][frame_index])
        self.line_ref_cons.set_data(x, self.results["reference"][frame_index])
        self.line_cons.set_data(x, self.results["conservative"][frame_index])

        mask_noncons = np.ones_like(self.results["nonconservative"], dtype=bool)
        mask_noncons[:frame_index + 1, :] = False
        mask_cons = np.ones_like(self.results["conservative"], dtype=bool)
        mask_cons[:frame_index + 1, :] = False

        self.img_noncons.set_data(np.ma.array(self.results["nonconservative"], mask=mask_noncons))
        self.img_cons.set_data(np.ma.array(self.results["conservative"], mask=mask_cons))

        self.time_line_noncons.set_ydata([t_value, t_value])
        self.time_line_cons.set_ydata([t_value, t_value])
        self.time_badge.set_text(f"t = {t_value:.2f}")
        self.progress_bar.set_width(0.89 * max(progress, 0.002))
        self.time_var.set(f"t = {t_value:.2f} / {t_end:.2f}")
        self.canvas.draw_idle()

    def _on_slider_changed(self, value):
        if self.results is None:
            return
        self.stop_playback(update_button=False)
        self.play_button.configure(text="Play")
        self.update_frame(int(float(value)))

    def toggle_playback(self):
        if self.results is None:
            messagebox.showinfo("No Data Yet", "Please compute the solution first.")
            return
        if self.playing:
            self.stop_playback()
            self.play_button.configure(text="Play")
        else:
            self.playing = True
            self.play_button.configure(text="Pause")
            self._playback_tick()

    def _playback_tick(self):
        if not self.playing or self.results is None:
            return

        next_frame = self.current_frame + 1
        if next_frame >= len(self.results["times"]):
            self.stop_playback(update_button=False)
            self.play_button.configure(text="Play")
            return

        self.slider_var.set(next_frame)
        self.update_frame(next_frame)
        delay_ms = max(18, int(45 / max(self.speed_var.get(), 0.25)))
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
        if self.results is None:
            return
        self.slider_var.set(0.0)
        self.update_frame(0)


def run_verification():
    config = BurgersConfig()
    results = build_results(
        config=config,
        expression=PRESETS["Triangular pulse (analytic reference)"]["expression"],
        preset_name="Triangular pulse (analytic reference)",
    )
    print("Verification summary:")
    print(results["reference_label"])
    print(f"Non-conservative L1 = {results['metrics']['noncons_l1']:.4e}")
    print(f"Conservative     L1 = {results['metrics']['cons_l1']:.4e}")


def main():
    parser = argparse.ArgumentParser(description="UI for Burgers equation conservative vs non-conservative comparison.")
    parser.add_argument("--verify", action="store_true", help="Run a lightweight non-UI verification.")
    args = parser.parse_args()

    if args.verify:
        run_verification()
        return

    root = tk.Tk()
    EulerEqUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
