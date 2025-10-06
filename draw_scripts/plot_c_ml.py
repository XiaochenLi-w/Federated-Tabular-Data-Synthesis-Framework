#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns

#plt.style.use("seaborn-whitegrid")
sns.set_style("whitegrid")
# -------------------------------------------------------
color_list = sns.color_palette("deep", 8)
fig = plt.figure(figsize=(9, 3))

#--------Fedprivsyn----------

y005_ndcg = [0.747434058, 0.700961856, 0.638155733, 0.565884087, 0.515403896]

# #--------Fedprivsyn_adv--------------
y005_re = [0.720833207, 0.657273103, 0.648670285, 0.607118536, 0.510260739]

eps_list = [5, 10, 15, 20, 25]

#--------------Fedprivsyn-------------------
ax1 = plt.subplot(1, 2, 1)
ax1.set_ylabel("ML Efficiency", fontsize=14)
# ax.set_ylabel("RE", fontsize=14)
ax1.set_xlabel('Number of Parties c', fontsize=14)
ax1.set_xticks(eps_list)
ax1.tick_params(axis="both", labelsize=13)
ax1.set_title('FedPrivSyn')

l1 = ax1.plot(eps_list,
              y005_ndcg,
              label="1-way",
              color=color_list[0],
              linestyle="-",
              marker="x",
              markersize=8,
              markerfacecolor='none')


#--------------Fedprivsyn_adv------------------
ax2 = plt.subplot(1, 2, 2)
ax2.set_ylabel("ML Efficiency", fontsize=14)
# ax.set_ylabel("RE", fontsize=14)
ax2.set_xlabel('Number of Parties c', fontsize=14)
ax2.set_xticks(eps_list)
ax2.tick_params(axis="both", labelsize=13)
ax2.set_title('AdaFedPrivSyn')

l1 = ax2.plot(eps_list,
              y005_re,
              label="1-way",
              color=color_list[0],
              linestyle="-",
              marker="x",
              markersize=8,
              markerfacecolor='none')

fig.tight_layout()
fig.subplots_adjust(left=0.082, bottom=0.152, right=0.975, top=0.756, wspace=0.356, hspace=0.2)

plt.show()
fig.savefig('D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/fig/vary_c_ml.pdf')