import argparse
import math
import re
import os
import tempfile
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


def find_recent_logs(count=1, job_outputs_dir=Path("jobOutputs")):
    if count < 1:
        raise ValueError("--num-logs must be at least 1")

    log_files = sorted(
        job_outputs_dir.glob("runJogo-*.out"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not log_files:
        raise FileNotFoundError(f"No runJogo-*.out files found in {job_outputs_dir}")
    return log_files[:count]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MatterGen training losses.")
    parser.add_argument(
        "log_files",
        nargs="*",
        help=(
            "Log file(s) to plot. Defaults to the newest jobOutputs/runJogo-*.out. "
            "Explicit files are plotted in the order given."
        ),
    )
    parser.add_argument(
        "-n",
        "--num-logs",
        type=int,
        default=1,
        help=(
            "Number of newest jobOutputs/runJogo-*.out files to concatenate, "
            "from least recent to most recent. Ignored when log files are provided."
        ),
    )
    parser.add_argument(
        "--output-png",
        default=None,
        help="Output PNG path. Defaults to jobOutputs/<log-name>.png.",
    )
    parser.add_argument("--window", type=int, default=10, help="Moving-median window.")
    parser.add_argument(
        "--validation-style",
        choices=("points", "lines"),
        default="points",
        help=(
            "How to draw validation values. 'points' keeps the original dot style; "
            "'lines' draws raw validation lines plus a frequency-scaled moving median."
        ),
    )
    args = parser.parse_args()
    if args.num_logs < 1:
        parser.error("--num-logs must be at least 1")
    return args


args = parse_args()
log_files = args.log_files or [str(path) for path in reversed(find_recent_logs(args.num_logs))]
output_png = args.output_png
window = args.window
validation_style = args.validation_style

number = r"[-+]?(?:nan|inf|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"

train_pattern = re.compile(
    r"epoch=(?P<epoch>\d+)\s+"
    r"step=(?P<step>\d+)\s+"
    r"global_step=(?P<global_step>\d+)\s+"
    rf"elapsed_train_seconds=(?P<elapsed_train_seconds>{number})\s+"
    rf"lr=(?P<lr>{number})\s+"
    rf"loss_train=(?P<loss_train>{number})\s+"
    rf"pos_train=(?P<pos_train>{number})\s+"
    rf"cell_train=(?P<cell_train>{number})\s+"
    rf"atom_train=(?P<atom_train>{number})",
    re.IGNORECASE,
)

val_pattern = re.compile(
    r"epoch=(?P<epoch>\d+)\s+"
    rf"lr=(?P<lr>{number})\s+"
    rf"loss_train=(?P<loss_train>{number})\s+"
    rf"loss_val=(?P<loss_val>{number})\s+"
    rf"pos_val=(?P<pos_val>{number})\s+"
    rf"cell_val=(?P<cell_val>{number})\s+"
    rf"atom_val=(?P<atom_val>{number})",
    re.IGNORECASE,
)

bp_config_pattern = re.compile(
    r"^\s*(?:root_dir|data_path):\s*(?P<path>[^#\s]+\.bp)\b"
)
bp_path_pattern = re.compile(r"(?P<path>[^#\s]+\.bp)\b")


def training_bp_basenames(log_files):
    basenames = []
    seen = set()
    fallback_paths = []
    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                match = bp_config_pattern.search(line)
                if match is None:
                    fallback = bp_path_pattern.search(line)
                    if fallback is not None:
                        fallback_paths.append(fallback.group("path"))
                    continue

                basename = Path(match.group("path").strip("'\"")).name
                if basename not in seen:
                    basenames.append(basename)
                    seen.add(basename)

    if basenames:
        return basenames

    for path in fallback_paths:
        basename = Path(path.strip("'\"")).name
        if basename not in seen:
            basenames.append(basename)
            seen.add(basename)
    return basenames


bp_basenames = training_bp_basenames(log_files)

epochs = []
global_steps = []
loss_train_steps = []
pos_train_steps = []
cell_train_steps = []
atom_train_steps = []

val_steps = []
loss_val = []
pos_val = []
cell_val = []
atom_val = []

last_train_global_step = None

for log_file in log_files:
    epoch_to_last_global_step = {}
    with open(log_file, "r") as f:
        for line in f:
            m = train_pattern.search(line)
            if m:
                epoch = int(m.group("epoch"))
                global_step = int(m.group("global_step"))
                epochs.append(epoch)
                global_steps.append(global_step)
                loss_train_steps.append(float(m.group("loss_train")))
                pos_train_steps.append(float(m.group("pos_train")))
                cell_train_steps.append(float(m.group("cell_train")))
                atom_train_steps.append(float(m.group("atom_train")))
                epoch_to_last_global_step[epoch] = global_step
                last_train_global_step = global_step
                continue

            m = val_pattern.search(line)
            if m:
                epoch = int(m.group("epoch"))
                val_steps.append(
                    epoch_to_last_global_step.get(
                        epoch,
                        last_train_global_step
                        if last_train_global_step is not None
                        else len(global_steps),
                    )
                )
                loss_val.append(float(m.group("loss_val")))
                pos_val.append(float(m.group("pos_val")))
                cell_val.append(float(m.group("cell_val")))
                atom_val.append(float(m.group("atom_val")))

if not global_steps:
    raise ValueError("No matching per-step training lines found in the provided log files.")

epochs = np.array(epochs)
global_steps = np.array(global_steps)
loss_train_steps = np.array(loss_train_steps)
pos_train_steps = np.array(pos_train_steps)
cell_train_steps = np.array(cell_train_steps)
atom_train_steps = np.array(atom_train_steps)

val_steps = np.array(val_steps)
loss_val = np.array(loss_val)
pos_val = np.array(pos_val)
cell_val = np.array(cell_val)
atom_val = np.array(atom_val)
x_steps = global_steps


def output_path_for_logs(log_files):
    job_outputs_dir = Path("jobOutputs")
    first_log = Path(log_files[0])
    if first_log.parent.name == "jobOutputs":
        job_outputs_dir = first_log.parent

    if len(log_files) == 1:
        log_label = first_log.stem
    else:
        log_label = f"{first_log.stem}-plus-{len(log_files) - 1}-logs"

    return job_outputs_dir / f"{log_label}.png"


if output_png is None:
    output_png = output_path_for_logs(log_files)
else:
    output_png = Path(output_png)

output_png.parent.mkdir(parents=True, exist_ok=True)


def moving_median(x, y, window):
    window = max(1, int(window))
    if len(y) < window:
        return x, y
    if window == 1:
        return x, y
    windows = np.lib.stride_tricks.sliding_window_view(y, window_shape=window)
    y_median = np.nanmedian(windows, axis=1)
    x_median = x[window - 1:]
    return x_median, y_median


def frequency_scaled_validation_window(train_x, val_x, train_window):
    train_window = max(1, int(train_window))
    if len(train_x) < 2 or len(val_x) < 2:
        return train_window

    train_spacing = np.diff(np.sort(train_x))
    val_spacing = np.diff(np.sort(val_x))
    train_spacing = train_spacing[train_spacing > 0]
    val_spacing = val_spacing[val_spacing > 0]
    if len(train_spacing) == 0 or len(val_spacing) == 0:
        return train_window

    frequency_factor = np.median(val_spacing) / np.median(train_spacing)
    if not np.isfinite(frequency_factor) or frequency_factor <= 0:
        return train_window

    return max(1, int(round(train_window / frequency_factor)))


def shifted_hue(color, hue_shift=0.055, min_value=0.35):
    rgb = np.array(mcolors.to_rgb(color))
    if np.max(rgb) - np.min(rgb) < 0.01:
        return mcolors.to_hex(np.clip(rgb + min_value, 0, 1))

    hsv = mcolors.rgb_to_hsv(rgb.reshape(1, 1, 3))[0, 0]
    hsv[0] = (hsv[0] + hue_shift) % 1.0
    hsv[1] = min(1.0, hsv[1] * 0.9)
    hsv[2] = min(1.0, max(hsv[2], 0.35) * 1.05)
    return mcolors.to_hex(mcolors.hsv_to_rgb(hsv.reshape(1, 1, 3))[0, 0])


def positive_ylim(*series):
    vals = np.concatenate([np.asarray(s) for s in series if len(s)])
    vals_pos = vals[vals > 0]
    if len(vals_pos) == 0:
        return (1e-6, 1.0)
    ymin = np.min(vals_pos)
    ymax = min(np.max(vals), 1000.0)
    if ymin == ymax:
        ymin = ymin / 10.0
    return (ymin, ymax)


def plot_series(ax, x, y, color, label=None):
    x_ma, y_ma = moving_median(x, y, window)
    ax.plot(x, y, color=color, alpha=0.25, linewidth=1.2)
    ax.plot(x_ma, y_ma, color=color, alpha=1.0, linewidth=2.0, label=label)


def values_between_steps(x, y, start_step, end_step):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > start_step)
        & (x <= end_step)
    )
    return y[mask]


def xy_between_steps(x, y, start_step, end_step):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (y > 0)
        & (x > start_step)
        & (x <= end_step)
    )
    return x[mask], y[mask]


def padded_log_bounds(values):
    values = np.asarray(values)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return None

    ymin = np.nanmin(values)
    ymax = np.nanmax(values)
    if ymin == ymax:
        return ymin / 1.12, ymax * 1.12

    pad_factor = (ymax / ymin) ** 0.08
    return ymin / pad_factor, ymax * pad_factor


def add_comparison_window_boxes(ax, x, y, color):
    latest_step = np.nanmax(x)
    windows = (
        (latest_step - 130000, latest_step - 100000, color),
        (latest_step - 30000, latest_step, "tab:orange"),
    )

    axis_ymin, axis_ymax = ax.get_ylim()
    axis_pad_factor = (axis_ymax / axis_ymin) ** 0.01
    visible_ymin = axis_ymin * axis_pad_factor
    visible_ymax = axis_ymax / axis_pad_factor

    for start_step, end_step, edgecolor in windows:
        _, y_window = xy_between_steps(x, y, start_step, end_step)
        y_bounds = padded_log_bounds(y_window)
        if y_bounds is None:
            continue

        ymin, ymax = y_bounds
        ymin = max(ymin, visible_ymin)
        ymax = min(ymax, visible_ymax)
        if ymin >= ymax:
            continue

        rect = Rectangle(
            (start_step, ymin),
            end_step - start_step,
            ymax - ymin,
            fill=False,
            edgecolor=edgecolor,
            linewidth=1.4,
            alpha=0.95,
            zorder=6,
        )
        ax.add_patch(rect)


def kde_curve(values, grid):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None

    std = np.std(values, ddof=1)
    q75, q25 = np.percentile(values, [75, 25])
    iqr_sigma = (q75 - q25) / 1.349 if q75 > q25 else 0.0
    sigma = min(std, iqr_sigma) if iqr_sigma > 0 else std
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = (np.max(values) - np.min(values)) / 6.0
    if not np.isfinite(sigma) or sigma <= 0:
        return None

    bandwidth = 0.9 * sigma * (len(values) ** (-1 / 5))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return None

    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z * z).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def one_sided_mean_decrease_pvalue(recent, previous):
    recent = np.asarray(recent)
    previous = np.asarray(previous)
    recent = recent[np.isfinite(recent)]
    previous = previous[np.isfinite(previous)]
    if len(recent) < 2 or len(previous) < 2:
        return None

    mean_diff = float(np.mean(recent) - np.mean(previous))
    recent_var = float(np.var(recent, ddof=1))
    previous_var = float(np.var(previous, ddof=1))
    standard_error = np.sqrt(recent_var / len(recent) + previous_var / len(previous))
    if not np.isfinite(standard_error) or standard_error <= 0:
        return None

    t_stat = mean_diff / standard_error
    if scipy_stats is not None:
        denominator = (
            (recent_var / len(recent)) ** 2 / (len(recent) - 1)
            + (previous_var / len(previous)) ** 2 / (len(previous) - 1)
        )
        if not np.isfinite(denominator) or denominator <= 0:
            return None
        degrees_of_freedom = (
            (recent_var / len(recent) + previous_var / len(previous)) ** 2 / denominator
        )
        p_value = float(scipy_stats.t.cdf(t_stat, degrees_of_freedom))
    else:
        p_value = float(normal_cdf(t_stat))

    if not np.isfinite(p_value):
        return None
    return max(0.0, min(1.0, p_value))


def format_pvalue(p_value):
    if p_value is None:
        return "p=n/a"
    if p_value < 1e-4:
        return f"p={p_value:.1e}"
    return f"p={p_value:.4f}"


def add_distribution_inset(ax, x, y, color):
    latest_step = np.nanmax(x)
    recent = values_between_steps(x, y, latest_step - 30000, latest_step)
    previous = values_between_steps(x, y, latest_step - 130000, latest_step - 100000)
    recent = recent[np.isfinite(recent)]
    previous = previous[np.isfinite(previous)]
    p_value = one_sided_mean_decrease_pvalue(recent, previous)

    inset = ax.inset_axes([0.08, 0.67, 0.25, 0.25])
    inset.patch.set_alpha(0.72)
    inset.set_title(format_pvalue(p_value), fontsize=6, pad=1.0)

    if len(recent) == 0 and len(previous) == 0:
        inset.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=5)
        inset.set_xticks([])
        inset.set_yticks([])
        return

    pooled = np.concatenate([vals for vals in (recent, previous) if len(vals)])
    lo, hi = np.nanpercentile(pooled, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = np.nanmin(pooled)
        hi = np.nanmax(pooled)
    if lo == hi:
        lo -= abs(lo) * 0.05 + 1e-12
        hi += abs(hi) * 0.05 + 1e-12
    pad = 0.08 * (hi - lo)
    lo -= pad
    hi += pad
    grid = np.linspace(lo, hi, 120)
    bins = np.linspace(lo, hi, 18)

    for vals, vals_color, label in (
        (previous, color, "-130k"),
        (recent, "tab:orange", "last 30k"),
    ):
        if len(vals) == 0:
            continue
        inset.hist(
            vals,
            bins=bins,
            density=True,
            color=vals_color,
            alpha=0.18,
            edgecolor=vals_color,
            linewidth=0.4,
        )
        density = kde_curve(vals, grid)
        if density is not None:
            inset.plot(grid, density, color=vals_color, linewidth=0.8, label=label)

    inset.set_xlim(lo, hi)
    inset.set_yticks([])
    inset.tick_params(axis="x", labelsize=4, length=1.5, pad=1)
    inset.tick_params(axis="y", length=0)
    for spine in inset.spines.values():
        spine.set_alpha(0.45)


validation_window = frequency_scaled_validation_window(x_steps, val_steps, window)


def plot_validation_series(ax, x, y, train_color, point_color, label=None):
    if validation_style == "points":
        ax.scatter(x, y, color=point_color, s=22, label=label, zorder=5)
        return

    color = shifted_hue(train_color)
    x_ma, y_ma = moving_median(x, y, validation_window)
    ax.plot(x, y, color=color, alpha=0.35, linewidth=1.3)
    ax.plot(
        x_ma,
        y_ma,
        color=color,
        alpha=1.0,
        linewidth=2.1,
        label=f"{label} median ({validation_window})" if label else None,
    )


fig, axes = plt.subplots(2, 2, figsize=(9, 8))
if bp_basenames:
    fig.suptitle(", ".join(bp_basenames), fontsize=13, fontweight="bold", y=0.998)
(ax1, ax2), (ax3, ax4) = axes

# Loss subplot
plot_series(ax1, x_steps, loss_train_steps, color="black", label="loss_train/row")
add_distribution_inset(ax1, x_steps, loss_train_steps, color="black")
if len(loss_val):
    plot_validation_series(
        ax1,
        val_steps,
        loss_val,
        train_color="black",
        point_color="red",
        label="loss_val/epoch",
    )
ax1.set_title("Loss per Concatenated Row")
ax1.set_xlabel("Global step")
ax1.set_ylabel("Value")
ax1.set_yscale("log")
ax1.set_ylim(*positive_ylim(loss_train_steps, loss_val))
add_comparison_window_boxes(ax1, x_steps, loss_train_steps, color="black")
ax1.grid(True, which="both", linestyle="--", alpha=0.5)
ax1.legend()

# pos_train subplot
plot_series(ax2, x_steps, pos_train_steps, color="blue", label="pos_train/row")
add_distribution_inset(ax2, x_steps, pos_train_steps, color="blue")
if len(pos_val):
    plot_validation_series(
        ax2,
        val_steps,
        pos_val,
        train_color="blue",
        point_color="navy",
        label="pos_val/epoch",
    )
ax2.set_title("pos per Concatenated Row")
ax2.set_xlabel("Global step")
ax2.set_ylabel("Value")
ax2.set_yscale("log")
ax2.set_ylim(*positive_ylim(pos_train_steps, pos_val))
add_comparison_window_boxes(ax2, x_steps, pos_train_steps, color="blue")
#ax2.set_ylim([0.5,1.0])
ax2.grid(True, which="both", linestyle="--", alpha=0.5)
ax2.legend()

# cell_train subplot
plot_series(ax3, x_steps, cell_train_steps, color="green", label="cell_train/row")
add_distribution_inset(ax3, x_steps, cell_train_steps, color="green")
if len(cell_val):
    plot_validation_series(
        ax3,
        val_steps,
        cell_val,
        train_color="green",
        point_color="darkgreen",
        label="cell_val/epoch",
    )
ax3.set_title("cell per Concatenated Row")
ax3.set_xlabel("Global step")
ax3.set_ylabel("Value")
ax3.set_yscale("log")
ax3.set_ylim(*positive_ylim(cell_train_steps, cell_val))
add_comparison_window_boxes(ax3, x_steps, cell_train_steps, color="green")
ax3.grid(True, which="both", linestyle="--", alpha=0.5)
ax3.legend()

# atom_train subplot
plot_series(ax4, x_steps, atom_train_steps, color="purple", label="atom_train/row")
add_distribution_inset(ax4, x_steps, atom_train_steps, color="purple")
if len(atom_val):
    plot_validation_series(
        ax4,
        val_steps,
        atom_val,
        train_color="purple",
        point_color="indigo",
        label="atom_val/epoch",
    )
ax4.set_title("atom per Concatenated Row")
ax4.set_xlabel("Global step")
ax4.set_ylabel("Value")
ax4.set_yscale("log")
ax4.set_ylim(*positive_ylim(atom_train_steps, atom_val))
add_comparison_window_boxes(ax4, x_steps, atom_train_steps, color="purple")
ax4.grid(True, which="both", linestyle="--", alpha=0.5)
ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.99] if bp_basenames else None)
plt.savefig(output_png, dpi=200, bbox_inches="tight")
plt.close()

print(f"Read {len(log_files)} log files")
print("Concatenated log order:")
for log_file in log_files:
    print(f"  {log_file}")
print(f"Parsed {len(global_steps)} training-step entries")
print(f"Parsed {len(val_steps)} validation-summary entries")
if validation_style == "lines":
    print(f"Validation moving-median window: {validation_window}")
print(f"Saved plot to {output_png}")
