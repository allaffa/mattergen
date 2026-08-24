import re
import numpy as np
import matplotlib.pyplot as plt

# ---- user settings ----
log_files = [
    "/lustre/orion/lrn070/proj-shared/patxi/jaime/mattergen/mattergen-5030024.out",
    "/lustre/orion/lrn070/proj-shared/patxi/jaime/mattergen/mattergen-5038269.out",
    "/lustre/orion/lrn070/proj-shared/patxi/jaime/mattergen/mattergen-5039238.out",
    "/lustre/orion/lrn070/proj-shared/patxi/jaime/mattergen/mattergen-5039666.out",
    "/lustre/orion/lrn070/proj-shared/patxi/jaime/mattergen/mattergen-5042965.out",
    "/lustre/orion/lrn070/proj-shared/patxi/jaime/mattergen/mattergen-5043368.out",
]
output_png = "loss.png"
window = 50

pattern = re.compile(
    r"epoch=(?P<epoch>\d+)\s+"
    r"lr=(?P<lr>[0-9.eE+-]+)\s+"
    r"loss_train=(?P<loss_train>[0-9.eE+-]+)\s+"
    r"loss_val=(?P<loss_val>[0-9.eE+-]+)\s+"
    r"pos_val=(?P<pos_val>[0-9.eE+-]+)\s+"
    r"cell_val=(?P<cell_val>[0-9.eE+-]+)\s+"
    r"atom_val=(?P<atom_val>[0-9.eE+-]+)"
)

epochs = []
loss_train = []
loss_val = []
pos_val = []
cell_val = []
atom_val = []

for log_file in log_files:
    with open(log_file, "r") as f:
        for line in f:
            if "step=" in line:
                continue

            m = pattern.search(line)
            if m:
                epochs.append(int(m.group("epoch")))
                loss_train.append(float(m.group("loss_train")))
                loss_val.append(float(m.group("loss_val")))
                pos_val.append(float(m.group("pos_val")))
                cell_val.append(float(m.group("cell_val")))
                atom_val.append(float(m.group("atom_val")))

if not epochs:
    raise ValueError("No matching validation lines found in the provided log files.")

epochs = np.array(epochs)
loss_train = np.array(loss_train)
loss_val = np.array(loss_val)
pos_val = np.array(pos_val)
cell_val = np.array(cell_val)
atom_val = np.array(atom_val)

def moving_average(x, y, window):
    if len(y) < window:
        return x, y
    kernel = np.ones(window) / window
    y_ma = np.convolve(y, kernel, mode="valid")
    x_ma = x[window - 1:]
    return x_ma, y_ma

def positive_ylim(*series):
    vals = np.concatenate([np.asarray(s) for s in series])
    vals_pos = vals[vals > 0]
    if len(vals_pos) == 0:
        return (1e-6, 1.0)
    ymin = np.min(vals_pos)
    ymax = min(np.max(vals), 10.0)
    if ymin == ymax:
        ymin = ymin / 10.0
    return (ymin, ymax)

def plot_series(ax, x, y, color, label=None):
    x_ma, y_ma = moving_average(x, y, window)
    ax.plot(x, y, color=color, alpha=0.25, linewidth=1.2)
    ax.plot(x_ma, y_ma, color=color, alpha=1.0, linewidth=2.0, label=label)

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
(ax1, ax2), (ax3, ax4) = axes

# Loss subplot
plot_series(ax1, epochs, loss_train, color="black", label="loss_train")
plot_series(ax1, epochs, loss_val, color="red", label="loss_val")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Value")
ax1.set_yscale("log")
ax1.set_ylim(*positive_ylim(loss_train, loss_val))
ax1.grid(True, which="both", linestyle="--", alpha=0.5)
ax1.legend()

# pos_val subplot
plot_series(ax2, epochs, pos_val, color="blue", label="pos_val")
ax2.set_title("pos_val")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Value")
ax2.set_yscale("log")
ax2.set_ylim(*positive_ylim(pos_val))
ax2.grid(True, which="both", linestyle="--", alpha=0.5)

# cell_val subplot
plot_series(ax3, epochs, cell_val, color="green", label="cell_val")
ax3.set_title("cell_val")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Value")
ax3.set_yscale("log")
ax3.set_ylim(*positive_ylim(cell_val))
ax3.grid(True, which="both", linestyle="--", alpha=0.5)

# atom_val subplot
plot_series(ax4, epochs, atom_val, color="purple", label="atom_val")
ax4.set_title("atom_val")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Value")
ax4.set_yscale("log")
ax4.set_ylim(*positive_ylim(atom_val))
ax4.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(output_png, dpi=200, bbox_inches="tight")
plt.close()

print(f"Read {len(log_files)} log files")
print(f"Parsed {len(epochs)} validation entries")
print(f"Saved plot to {output_png}")
