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

# Ldp_ndcg = [0.5981, 0.6875, 0.7474]
# Direct_ndcg = [0.7177, 0.7875, 0.7539]
Ldp_ndcg = [0.8665, 0.9040, 0.7735070595686117]
Direct_ndcg = [0.8674, 0.9227, 0.7653020511222647]
label_dist = [0.8614, 0.9138, 0.7799567889753306]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 3

l1 = ax1.bar(x, Ldp_ndcg,  width=width, color=color_list[0], label='Label')
l2 = ax1.bar(x + width, Direct_ndcg, width=width, color=color_list[1], label='Random')
l3 = ax1.bar(x + 2 * width, label_dist, width=width, color=color_list[2], label='Uniform')
plt.xticks(x + 0.25, x_label, rotation=0)
ax1.set_ylabel("Accuracy", fontsize=14)
ax1.set_xlabel("Different evaluations", fontsize=14)
ax1.set_title('FedPrivSyn')
ax1.set_ylim(0, 1.0) 

ax2 = plt.subplot(1, 2, 2)

Ldp_re = [0.8646, 0.9225, 0.7500591771585322]
Direct_re = [0.8614, 0.9233, 0.7109134616262055]
label_dist = [0.8463, 0.9225, 0.7330640781934655]

l1 = ax2.bar(x, Ldp_ndcg,  width=width, color=color_list[0], label='Label')
l2 = ax2.bar(x + width, Direct_ndcg, width=width, color=color_list[1], label='Random')
l3 = ax2.bar(x + 2 * width, label_dist, width=width, color=color_list[2], label='Uniform')
plt.xticks(x + 0.25, x_label, rotation=0)
ax2.set_ylabel("Accuracy", fontsize=14)
ax2.set_xlabel("Different evaluations", fontsize=14)
ax2.set_title('AdaFedPrivSyn')
ax2.set_ylim(0, 1.0) 

# fig.legend(loc='center', bbox_to_anchor=(0.25, 0.78), ncol=1, prop={'size': 10}, frameon=True, edgecolor='gray')
legend_list = ['Biased Distributed', 'Random Distributed', 'Uniform Distributed']
fig.legend([l1, l2, l3], labels=legend_list, loc='upper center', bbox_to_anchor=(0.5, 0.99),
           ncol=4, prop={'size': 12}, frameon=True, edgecolor='gray')
fig.tight_layout()
fig.subplots_adjust(left=0.076, bottom=0.147, right=0.96, top=0.777, wspace=0.236, hspace=0.2)

plt.show()

fig.savefig('./draw_scripts/fig/distribution.pdf')