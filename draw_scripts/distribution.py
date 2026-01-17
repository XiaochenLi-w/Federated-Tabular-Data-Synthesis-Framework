#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#plt.style.use("seaborn-whitegrid")
sns.set_style("whitegrid")
# -------------------------------------------------------
color_list = sns.color_palette("deep", 8)
fig = plt.figure(figsize=(10, 3))

ax1 = plt.subplot(1, 2, 1)
size = 3
x_label = ['Query', 'Fidelity', 'ML']
x = np.arange(size)

Ldp_ndcg = [0.5981, 0.6875, 0.7474]
Direct_ndcg = [0.7177, 0.7875, 0.7539]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

l1 = ax1.bar(x, Ldp_ndcg,  width=width, color=color_list[0], label='BasicLdpPD')
l2 = ax1.bar(x + width, Direct_ndcg, width=width, color=color_list[1], label='DirectLdpPD')
plt.xticks(x + 0.25, x_label, rotation=0)
ax1.set_ylabel("Accuracy", fontsize=14)
ax1.set_xlabel("Different evaluations", fontsize=12)
ax1.set_title('FedPrivSyn')
ax1.set_ylim(0, 1.0) 

ax2 = plt.subplot(1, 2, 2)

Ldp_re = [0.7648, 0.8434, 0.720]
Direct_re = [0.8291, 0.9079, 0.7257]

ax2.bar(x, Ldp_re,  width=width, color=color_list[0], label='BasicLdpPD')
ax2.bar(x + width, Direct_re, width=width, color=color_list[1], label='DirectLdpPD' )
plt.xticks(x + 0.25, x_label, rotation=0)
ax2.set_ylabel("Accuracy", fontsize=14)
ax2.set_xlabel("Different evaluations", fontsize=12)
ax2.set_title('AdaFedPrivSyn')
ax2.set_ylim(0, 1.0) 

# fig.legend(loc='center', bbox_to_anchor=(0.25, 0.78), ncol=1, prop={'size': 10}, frameon=True, edgecolor='gray')
legend_list = ['Uniform Distributed', 'Random Distributed']
fig.legend([l1, l2], labels=legend_list, loc='upper center', bbox_to_anchor=(0.5, 0.99),
           ncol=4, prop={'size': 10}, frameon=True, edgecolor='gray')
fig.tight_layout()
fig.subplots_adjust(left=0.076, bottom=0.147, right=0.96, top=0.777, wspace=0.236, hspace=0.2)

plt.show()

fig.savefig('D:/Fed_Privsyn/Tabular-Data-Synthesis-Framework/draw_scripts/fig/distribution.pdf')