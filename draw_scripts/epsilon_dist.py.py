import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------
# Style
# ---------------------------
sns.set_style("whitegrid")

# ---------------------------
# X-axis (categorical, equal spacing)
# ---------------------------
epsilons = ["0.2", "1", "5"]
x = np.arange(len(epsilons))

# ---------------------------
# Labels & styles
# ---------------------------
methods = [
    "FedPrivSyn (80%)",
    "FedPrivSyn (60%)",
    "FedPrivSyn (40%)",
    "AdaFedPrivSyn (80%)",
    "AdaFedPrivSyn (60%)",
    "AdaFedPrivSyn (40%)",
]

colors = ["tab:blue"] * 3 + ["tab:orange"] * 3
linestyles = ["-", "--", ":"] * 2
markers = ["o", "s", "^"] * 2

# ---------------------------
# Your data (replace with real values)
# shape: (6, 3)
# ---------------------------
metrics = [
    [[0.018, 0.018, 0.005],
    [0.040093198, 0.020015712, 0.005855622],
    [0.034691233, 0.019268233, 0.006845847],
    [0.017, 0.009, 0.006],
    [0.012075592, 0.008400481, 0.006359537],
    [0.022434464, 0.009392702, 0.005711449]],
    [[0.23, 0.281, 0.060825628],
    [0.543641463, 0.291312696, 0.074041281],
    [0.526536731, 0.269489571, 0.090128962],
    [0.158, 0.097, 0.033658642],
    [0.153749906, 0.072754373, 0.032995606],
    [0.273624904, 0.088451425, 0.047150743]],
    [[0.648, 0.736, 0.752],
    [0.645240722, 0.703644501, 0.72734914],
    [0.678287776, 0.713552542, 0.768550106],
    [0.609, 0.718, 0.728],
    [0.619289051, 0.643368486, 0.739806499],
    [0.655325222, 0.672860022, 0.740751291]]
]

ylabels = [
    "Query Error",
    "Fidelity Error",
    "ML Efficiency",
]

# ---------------------------
# Plot
# ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), sharex=True)

for ax, data, ylabel in zip(axes, metrics, ylabels):
    for i in range(6):
        ax.plot(
            x,
            data[i],
            label=methods[i],
            color=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2,
            markersize=6,
        )

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(epsilons)
    ax.set_xlabel(r"$\epsilon$")

# ---------------------------
# Shared legend
# ---------------------------
handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=6,
    frameon=True,
    edgecolor="gray",
    bbox_to_anchor=(0.5, 0.95),
    prop={'size': 12}
)

# ---------------------------
# Layout
# ---------------------------
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.subplots_adjust(left=0.05, bottom=0.146, right=0.98, top=0.813, wspace=0.211, hspace=0.2)
plt.show()
fig.savefig('/draw_scripts/fig/epsilon_dist.pdf')