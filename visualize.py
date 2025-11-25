import re
from collections import defaultdict
import matplotlib.pyplot as plt

def plot_metrics_from_log(
    log_path,
    metrics_to_plot=None,
    insert_newlines_before_epoch=True,
):
    """
    Parse a training log and plot metrics vs epoch.

    Parameters
    ----------
    log_path : str
        Path to the log file.
    metrics_to_plot : list[str] or None
        List of metric names to plot (e.g. ["loss_train_epoch", "pos_train_epoch"]).
        If None, all numeric metrics except 'v_num' are plotted.
    insert_newlines_before_epoch : bool
        If True, will treat every 'Epoch N:' as if it starts a new line.
        This is useful when logs get pasted with '...0Epoch 199:...' glued together.
    """

    # --- 1. Read log file ---
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Optional: fix copies where 'Epoch 198:' was glued onto previous text
    if insert_newlines_before_epoch:
        text = re.sub(r'(?!^)(Epoch\s+\d+:)', r'\n\1', text)

    # --- 2. Split into per-epoch chunks ---
    # Each chunk starts at 'Epoch N:' and goes until next 'Epoch M:' or end of text
    epoch_chunks = re.findall(
        r'(Epoch\s+\d+:.*?)(?=Epoch\s+\d+:|$)',
        text,
        flags=re.S
    )

    # epoch -> {metric_name: value}
    epoch_metrics = defaultdict(dict)

    for chunk in epoch_chunks:
        # Find epoch number
        m_epoch = re.search(r'Epoch\s+(\d+):', chunk)
        if not m_epoch:
            continue
        epoch = int(m_epoch.group(1))

        # Find all key=value pairs with numeric values
        # e.g. loss_train_step=0.843, v_num=3.9e+6
        for key, value in re.findall(r'([a-zA-Z0-9_]+)=([-\d\.eE+]+)', chunk):
            try:
                epoch_metrics[epoch][key] = float(value)
            except ValueError:
                # If for some reason it isn’t a float, skip it
                continue

    if not epoch_metrics:
        raise ValueError("No epochs/metrics found in the log file.")

    # --- 3. Collect metrics into arrays sorted by epoch ---
    epochs = sorted(epoch_metrics.keys())
    all_metric_names = set()
    for e in epochs:
        all_metric_names.update(epoch_metrics[e].keys())

    # By default, plot everything except v_num
    if metrics_to_plot is None:
        metrics_to_plot = sorted(m for m in all_metric_names if m != "v_num")

    # --- 4. Plot ---
    plt.figure(figsize=(10, 6))

    for metric in metrics_to_plot:
        ys = []
        for e in epochs:
            v = epoch_metrics[e].get(metric, None)
            ys.append(v)
        plt.plot(epochs, ys, marker="o", label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.yscale("log")
    plt.title("Training Metrics vs Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("metric.png")

    # Return the raw data too in case you want to inspect / save
    return {
        "epochs": epochs,
        "metrics": epoch_metrics,
    }

# Example usage
result = plot_metrics_from_log(
    "mattergen-3897713.out",
    metrics_to_plot=["loss_train_epoch", "pos_train_epoch", "cell_train_epoch"]
)

