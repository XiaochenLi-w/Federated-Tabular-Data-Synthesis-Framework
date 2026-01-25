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

# y005_ndcg = [0.104062644, 0.122228517, 0.096117662, 0.101490916, 0.11659348]
# y1_ndcg = [0.02569891, 0.038170172, 0.043800757, 0.042973591, 0.045326987]
# y25_ndcg = [0.006484825, 0.008944112, 0.010529249, 0.0107809, 0.010766109]

y005_ndcg = [0.022155483903986903, 0.04125927631915655, 0.03898495317058191, 0.06301161779006091, 0.07664291928962587]
y1_ndcg = [0.007369977992732483, 0.009930139720558883, 0.013563795485951173, 0.018038947745534568, 0.02131982189467219]
y25_ndcg = [0.0018968217411331183, 0.002427350427350427, 0.003274732586109831, 0.004944572393674189, 0.005753979221045089]

# #--------Fedprivsyn_adv--------------
# y005_re = [0.0511605, 0.09959046, 0.123498541, 0.165604688, 0.200328062]
# y1_re = [0.013404985, 0.029103741, 0.040323814, 0.047434106, 0.045864988]
# y25_re = [0.003640156, 0.007973694, 0.010220533, 0.010444393, 0.010664927]

y005_re = [0.00937453298531143, 0.016586979886381085, 0.020624545780234402, 0.02825180408413942, 0.032495880034802194]
y1_re = [0.007625415834996673, 0.007100158657044883, 0.00864609243052357, 0.0121407441527202, 0.011533241209887917]
y25_re = [0.0022338400122831263, 0.0021685347254209526, 0.0025696299708275754, 0.003447617585342136, 0.0032456113414197248]

eps_list = [5, 10, 15, 20, 25]

#--------------Fedprivsyn-------------------
ax1 = plt.subplot(1, 2, 1)
ax1.set_ylabel("Query Error", fontsize=14)
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

l2 = ax1.plot(eps_list,
              y1_ndcg,
              label="2-way",
              color=color_list[2],
              linestyle="-",
              marker="p",
              markersize=8,
              markerfacecolor='none')

l3 = ax1.plot(eps_list,
              y25_ndcg,
              label="3-way",
              color=color_list[3],
              linestyle="-",
              marker="o",
              markersize=8,
              markerfacecolor='none')


#--------------Fedprivsyn_adv------------------
ax2 = plt.subplot(1, 2, 2)
ax2.set_ylabel("Query Error", fontsize=14)
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

l2 = ax2.plot(eps_list,
              y1_re,
              label="2-way",
              color=color_list[2],
              linestyle="-",
              marker="p",
              markersize=8,
              markerfacecolor='none')

l3 = ax2.plot(eps_list,
              y25_re,
              label="3-way",
              color=color_list[3],
              linestyle="-",
              marker="o",
              markersize=8,
              markerfacecolor='none')


legend_list = ['1-way', '2-way', '3-way']
fig.legend([l1, l2, l3], labels=legend_list, loc='upper center', bbox_to_anchor=(0.5, 0.94),
           ncol=6, prop={'size': 12}, frameon=True, edgecolor='gray')
fig.tight_layout()
fig.subplots_adjust(left=0.100, bottom=0.160, right=0.975, top=0.700, wspace=0.300, hspace=0.2)

plt.show()
fig.savefig('./draw_scripts/fig/vary_c.pdf')