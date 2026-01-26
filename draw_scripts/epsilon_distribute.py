import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ===== 配置部分 =====
metrics = ["Query Error", "Fidelity Error", "ML Efficiency"]
methods = ["FedPrivSyn(80%)", "FedPrivSyn(40%)", "AdaFedPrivSyn(80%)", "AdaFedPrivSyn(40%)"]
epsilons = [0.2, 1, 5]

results = {
    "Query Error": {
        "FedPrivSyn(80%)": [0.054, 0.041, 0.018],
        "FedPrivSyn(40%)": [0.050380367, 0.040376887, 0.025],
        "AdaFedPrivSyn(80%)": [0.056, 0.029, 0.015],
        "AdaFedPrivSyn(40%)": [0.037228569, 0.028181227, 0.013],
    },
    "Fidelity Error": {
        "FedPrivSyn(80%)": [0.878, 0.534, 0.27],
        "FedPrivSyn(40%)": [0.535225964, 0.492590347, 0.375],
        "AdaFedPrivSyn(80%)": [0.837, 0.393, 0.203],
        "AdaFedPrivSyn(40%)": [0.475327123, 0.42201696, 0.17],
    },
    "ML Efficiency": {
        "FedPrivSyn(80%)": [0.424, 0.559, 0.753],
        "FedPrivSyn(40%)": [0.453116058, 0.640734783, 0.747],
        "AdaFedPrivSyn(80%)": [0.561, 0.593, 0.632],
        "AdaFedPrivSyn(40%)": [0.381308571, 0.58795565, 0.721],
    }
}

# ===== 转换为 DataFrame =====
data = []
for metric in metrics:
    for method in methods:
        for i, eps in enumerate(epsilons):
            value = results[metric][method][i]
            data.append([metric, method, eps, value])

df = pd.DataFrame(data, columns=["Metric", "Method", "Epsilon", "Value"])

# ===== 画图部分 =====
sns.set(style="whitegrid", font_scale=1)

fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4), sharey=True)

for i, metric in enumerate(metrics):
    ax = axes[i]
    subset = df[df["Metric"] == metric]
    sns.barplot(
        data=subset,
        y="Method", x="Value",
        hue="Epsilon",
        ax=ax,
        orient="h",
        palette="Set2"
    )
    ax.set_title(metric)
    # 修改横坐标名称
    if metric == "ML Efficiency":
        ax.set_xlabel("F1 Score")
    else:
        ax.set_xlabel("Error")
    ax.set_ylabel("Method" if i == 0 else "")

    ax.get_legend().remove()

# 统一图例并改成 epsilon=xxx 格式
handles, labels = axes[0].get_legend_handles_labels()
labels = [f"epsilon={l}" for l in labels]
fig.legend(handles, labels, loc="upper center", ncol=len(epsilons))

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()
fig.savefig('D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/fig/epsilon_dist.pdf')