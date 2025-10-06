# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from sklearn.decomposition import PCA

# def load_datasets(real_path, gen_path):
#     """读取两个 CSV 文件"""
#     df_real = pd.read_csv(real_path)
#     df_gen = pd.read_csv(gen_path)
#     return df_real, df_gen

# def sample_datasets(df_real, df_gen, n=1000, random_state=None):
#     """从两个数据集中各抽样 n 条"""
#     df_real_sample = df_real.sample(n=min(n, len(df_real)), random_state=random_state)
#     df_gen_sample  = df_gen.sample(n=min(n, len(df_gen)), random_state=random_state)
#     return df_real_sample, df_gen_sample

# def plot_pca_projection(df_real, df_gen, numeric_attrs, ax, title, color_list):
#     """多维数值属性 → PCA 降维散点图 (画在指定ax上)"""
#     # 抽样
#     df_real, df_gen = sample_datasets(df_real, df_gen, n=100, random_state=42)

#     # 拼接数据
#     df_real_copy = df_real[numeric_attrs].copy()
#     df_gen_copy = df_gen[numeric_attrs].copy()
#     df_real_copy["dataset"] = "Real"
#     df_gen_copy["dataset"] = "Generated"
#     df_all = pd.concat([df_real_copy, df_gen_copy], axis=0)

#     # PCA 降维
#     X = df_all[numeric_attrs].fillna(0).values
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)
#     df_all["pca1"] = X_pca[:,0]
#     df_all["pca2"] = X_pca[:,1]

#     # 绘制散点
#     sns.scatterplot(data=df_all, x="pca1", y="pca2", hue="dataset", 
#                     alpha=0.6, ax=ax, palette=color_list[:2])
#     ax.set_title(title)

# if __name__ == "__main__":
#     # ==== 数据路径 ====
#     real_path = "./dataset/insurance_org.csv"
#     gen_path_fed  = "./dataset/insurance_fed_privsyn.csv"
#     gen_path_adv  = "./dataset/insurance_adv_privsyn.csv"
#     gen_path_privsyn = "./dataset/insurance_privsyn.csv"

#     # 载入原始数据
#     df_real = pd.read_csv(real_path)

#     # 数值属性列
#     numeric_attrs = [col for col in df_real.columns if pd.api.types.is_numeric_dtype(df_real[col])]

#     # 颜色
#     color_list = sns.color_palette("deep", 8)

#     # 创建一行三列子图
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))

#     # PrivSyn
#     df_real, df_priv = load_datasets(real_path, gen_path_privsyn)
#     plot_pca_projection(df_real, df_priv, numeric_attrs, axes[0], "PrivSyn", color_list)

#     # FedPrivSyn
#     df_real, df_fed = load_datasets(real_path, gen_path_fed)
#     plot_pca_projection(df_real, df_fed, numeric_attrs, axes[1], "FedPrivSyn", color_list)

#     # AdvPrivSyn
#     df_real, df_adv = load_datasets(real_path, gen_path_adv)
#     plot_pca_projection(df_real, df_adv, numeric_attrs, axes[2], "AdvPrivSyn", color_list)


#     plt.tight_layout()
#     plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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

def plot_pca_projection(df_real, df_gen, numeric_attrs, pca, ax, title, color_list):
    """使用固定的 PCA 对象对两个数据集降维并绘制散点图"""
    # 抽样
    df_real, df_gen = sample_datasets(df_real, df_gen, n=100, random_state=42)

    # 拼接数据
    df_real_copy = df_real[numeric_attrs].copy()
    df_gen_copy = df_gen[numeric_attrs].copy()
    df_real_copy["dataset"] = "Real Dataset"
    df_gen_copy["dataset"] = "Synthetic Dataset"
    df_all = pd.concat([df_real_copy, df_gen_copy], axis=0)

    # 使用固定的 PCA 转换
    X = df_all[numeric_attrs].fillna(0).values
    X_pca = pca.transform(X)

    df_all["pca1"] = X_pca[:,0]
    df_all["pca2"] = X_pca[:,1]

    # 绘制散点
    sns.scatterplot(data=df_all, x="pca1", y="pca2", hue="dataset", 
                    alpha=0.6, ax=ax, palette=color_list[:2])
    ax.legend(loc="lower right")
    ax.set_title(title)

if __name__ == "__main__":
    # ==== 数据路径 ====
    real_path = "D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/dataset/insurance_org.csv"
    gen_path_fed  = "D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/dataset/insurance_fed_privsyn.csv"
    gen_path_adv  = "D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/dataset/insurance_adv_privsyn.csv"
    gen_path_privsyn = "D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/dataset/insurance_privsyn.csv"

    # 载入原始数据
    df_real = pd.read_csv(real_path)

    # 数值属性列
    numeric_attrs = [col for col in df_real.columns if pd.api.types.is_numeric_dtype(df_real[col])]

    # 用原始数据集拟合一次 PCA，固定主成分方向
    pca = PCA(n_components=2)
    pca.fit(df_real[numeric_attrs].fillna(0).values)

    # 颜色
    color_list = sns.color_palette("deep", 8)

    # 创建一行三列子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # PrivSyn
    df_real, df_priv = load_datasets(real_path, gen_path_privsyn)
    plot_pca_projection(df_real, df_priv, numeric_attrs, pca, axes[0], "PrivSyn", color_list)

    # FedPrivSyn
    df_real, df_fed = load_datasets(real_path, gen_path_fed)
    plot_pca_projection(df_real, df_fed, numeric_attrs, pca, axes[1], "FedPrivSyn", color_list)

    # AdvPrivSyn
    df_real, df_adv = load_datasets(real_path, gen_path_adv)
    plot_pca_projection(df_real, df_adv, numeric_attrs, pca, axes[2], "AdvPrivSyn", color_list)


    plt.tight_layout()
    plt.show()
    fig.savefig('D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/fig/pca.pdf')