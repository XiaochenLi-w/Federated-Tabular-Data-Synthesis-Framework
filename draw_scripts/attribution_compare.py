import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_datasets(real_path, gen_path):
    """读取两个 CSV 文件"""
    df_real = pd.read_csv(real_path)
    df_gen = pd.read_csv(gen_path)
    return df_real, df_gen

def sample_datasets(df_real, df_gen, n=1000, random_state=None):
    """从两个数据集中各抽样 n 条"""
    df_real_sample = df_real.sample(n=min(n, len(df_real)), random_state=random_state)
    df_gen_sample  = df_gen.sample(n=min(n, len(df_gen)), random_state=random_state)
    return df_real_sample, df_gen_sample

def plot_numeric_distribution(df_real, df_gen, attr, ax, title, method="kde"):
    """数值型属性的对比分布（画在指定子图 ax 上）"""
    df_real, df_gen = sample_datasets(df_real, df_gen, n=100, random_state=42)

    if method == "kde":
        sns.kdeplot(df_real[attr], label="Real Dataset", fill=True, alpha=0.3, ax=ax)
        sns.kdeplot(df_gen[attr], label="Synthetic Dataset", fill=True, alpha=0.3, ax=ax)
    elif method == "dot":
        ax.scatter(df_real[attr], np.zeros_like(df_real[attr])+0.1,
                   alpha=0.4, label="Real", s=20)
        ax.scatter(df_gen[attr], np.zeros_like(df_gen[attr])-0.1,
                   alpha=0.4, label="Generated", s=20)
        ax.set_yticks([])

    ax.set_title(title)
    ax.legend(loc="upper right")  # 固定图例在右下角

if __name__ == "__main__":
    # ==== 数据路径 ====
    real_path = "D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/dataset/insurance_org.csv"
    gen_path_fed  = "D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/dataset/insurance_fed_privsyn.csv"
    gen_path_adv  = "D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/dataset/insurance_adv_privsyn.csv"
    gen_path_privsyn = "D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/dataset/insurance_privsyn.csv"

    # 载入原始数据
    df_real = pd.read_csv(real_path)

    # 想对比的单个数值属性
    target_attr = "label"  

    # 创建一行三列子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    #PrivSyn
    df_real, df_priv = load_datasets(real_path, gen_path_privsyn)
    plot_numeric_distribution(df_real, df_priv, target_attr, axes[0], "PrivSyn", method="kde")

    #FedPrivSyn
    df_real, df_fed = load_datasets(real_path, gen_path_fed)
    plot_numeric_distribution(df_real, df_fed, target_attr, axes[1], "FedPrivSyn", method="kde")

    #AdvPrivSyn
    df_real, df_adv = load_datasets(real_path, gen_path_adv)
    plot_numeric_distribution(df_real, df_adv, target_attr, axes[2], "AdaFedPrivSyn", method="kde")

    plt.tight_layout()
    plt.show()
    fig.savefig('D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/fig/attribute_dist.pdf')