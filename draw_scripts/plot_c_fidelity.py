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

y005_ndcg = [0.374654339, 0.532836565, 0.574181382, 0.585152827, 0.617095851]

# #--------Fedprivsyn_adv--------------
y005_re = [0.170287675, 0.368040242, 0.613503117, 0.661294208, 0.653206908]

eps_list = [5, 10, 15, 20, 25]

#--------------Fedprivsyn-------------------
ax1 = plt.subplot(1, 2, 1)
ax1.set_ylabel("Fidelity Error", fontsize=14)
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
ax2.set_ylabel("Fidelity Error", fontsize=14)
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
fig.savefig('D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/fig/vary_c_fidelity.pdf')