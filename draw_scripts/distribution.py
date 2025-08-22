#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.style.use("seaborn-whitegrid")
# -------------------------------------------------------
color_list = sns.color_palette("deep", 8)
fig = plt.figure(figsize=(10, 3))

ax1 = plt.subplot(1, 2, 1)
size = 3
x_label = ['Query Error', 'Fidelity Error', 'ML Efficiency']
x = np.arange(size)

Ldp_ndcg = [0.0117, 0.1295, 0.177]
Direct_ndcg = [0.0447, 0.1187, 0.3119]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

l1 = ax1.bar(x, Ldp_ndcg,  width=width, color=color_list[0], label='BasicLdpPD')
l2 = ax1.bar(x + width, Direct_ndcg, width=width, color=color_list[1], label='DirectLdpPD')
plt.xticks(x + 0.25, x_label, rotation=0)
ax1.set_ylabel("NDCG", fontsize=14)
ax1.set_xlabel("Different Dataset for Warm-up Stage", fontsize=12)

ax2 = plt.subplot(1, 2, 2)

Ldp_re = [4717.2527, 4141.8252, 3946.981]
Direct_re = [4653.0409, 4303.1231, 3013.5462]
AdvLdp_re = [4995.7579, 3319.7496, 1322.1195]
BufferLdp_re = [5023.7352, 3541.8606, 1299.8992]

ax2.bar(x, Ldp_re,  width=width, color=color_list[0], label='BasicLdpPD')
ax2.bar(x + width, Direct_re, width=width, color=color_list[1], label='DirectLdpPD' )
plt.xticks(x + 0.25, x_label, rotation=0)
ax2.set_ylabel("AAE", fontsize=14)
ax2.set_xlabel("Different Dataset for Warm-up Stage", fontsize=12)

# fig.legend(loc='center', bbox_to_anchor=(0.25, 0.78), ncol=1, prop={'size': 10}, frameon=True, edgecolor='gray')
legend_list = ['Uniform Distributed', 'Random Distributed']
fig.legend([l1, l2], labels=legend_list, loc='upper center', bbox_to_anchor=(0.5, 0.99),
           ncol=4, prop={'size': 10}, frameon=True, edgecolor='gray')
fig.tight_layout()
fig.subplots_adjust(left=0.076, bottom=0.147, right=0.96, top=0.844, wspace=0.236, hspace=0.2)

plt.show()

#fig.savefig('C:/Users/xiaoc/Dropbox/应用/Overleaf/SIGMODpaperV2mod030/fig/experiment/warm_up.pdf')