# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import rcParams
import matplotlib.font_manager as font_manager
from matplotlib.patches import FancyBboxPatch, Patch
# rcParams['font.family'] = 'monospace'
# rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['text.usetex'] = True
plt.subplots_adjust(wspace=0.01)
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False




def rounded_bar(ax, x, height, width=0.8, align="center", facecolor='blue', edgecolor='black', linewidth=1, rounding_size=0.2):
    if align == "center":
        position = (x - width / 2, 0)
    elif align == "left":
        position = (x - width, 0)
    elif align == "right":
        position = (x, 0)
    else:
        raise ValueError("align should be 'center', 'left', or 'right'")

    bbox = FancyBboxPatch(
        position, width, height,
        boxstyle=f"round,pad=0.0, rounding_size={rounding_size}",
        linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor,)
    ax.add_patch(bbox)


fig, axs = plt.subplots(2, 2)

for axis in ['top','bottom','left','right']:
    axs[0, 0].spines[axis].set_linewidth(1.0)
for axis in ['top','bottom','left','right']:
    axs[0, 1].spines[axis].set_linewidth(1.0)

for axis in ['top','bottom','left','right']:
    axs[1, 0].spines[axis].set_linewidth(1.0)

for axis in ['top','bottom','left','right']:
    axs[1, 1].spines[axis].set_linewidth(1.0)


xticks = [1, 2, 3, 4, 5]
ops_depth = ["NYUv2", "ScanNet", "ETH3D", "KITTI", ""]

abs_single = [10.8, 11.4, 13.3, 16.7]
abs_multi = [9.8, 10.2, 11.7, 14.8]

delta1_single = [88.7, 87.9, 84.8, 77.2]
delta1_multi = [90.6, 90.4, 87.4, 81.2]

y_abs = [5.0, 20.0]
y_delta1 = [70.0, 100.0]

# axs[0,0].bar(ops_depth, abs_single, color='mistyrose', width=0.15, align='center', label='SingleTask')
# axs[0,0].bar(ops_depth, abs_multi, color='blue', width=0.15, align='edge', label='MultiTask')
# axs[0,0].grid(axis='y', linewidth=0.15, linestyle='--')
# axs[0,0].axes.set_xlim(-0.5, 1.5)
# axs[0,0].set_ylim([88.0, 90.5])

# single_abs_color = 'tab:blue'
# multi_abs_color = 'tab:purple'

single_abs_color = 'steelblue'
multi_abs_color = 'skyblue'

axs[0,0].set_ylim(y_abs)
for xi, yi in zip(xticks[:5], abs_single):
    rounded_bar(axs[0,0], xi, yi, width=0.3, align="left", facecolor=single_abs_color, edgecolor=single_abs_color, linewidth=0, rounding_size=0.25)

for xi, yi in zip(xticks[:5], abs_multi):
    rounded_bar(axs[0,0], xi, yi, width=0.3, align="center", facecolor=multi_abs_color, edgecolor=multi_abs_color, linewidth=0, rounding_size=0.25)

axs[0,0].set_xticks(xticks)
axs[0,0].set_xticklabels(ops_depth, fontsize=10)
axs[0,0].set_xlabel("(a) Depth (AbsRel$\downarrow$)", fontsize=12)
legend_patch = Patch(facecolor=single_abs_color, edgecolor=single_abs_color, label='Single-task')
legend_patch2 = Patch(facecolor=multi_abs_color, edgecolor=multi_abs_color, label='Multi-task')
axs[0,0].legend(handles=[legend_patch, legend_patch2], loc='best', fontsize=10)


# single_delta1_color = 'plum'
# multi_delta1_color = 'violet'
single_delta1_color = 'lightskyblue'
multi_delta1_color = 'dodgerblue'

axs[0,1].set_ylim(y_delta1)
for xi, yi in zip(xticks[:5], delta1_single):
    rounded_bar(axs[0,1], xi, yi, width=0.3, align="left", facecolor=single_delta1_color, edgecolor=single_delta1_color, linewidth=0, rounding_size=0.25)

for xi, yi in zip(xticks[:5], delta1_multi):
    rounded_bar(axs[0,1], xi, yi, width=0.3, align="center", facecolor=multi_delta1_color, edgecolor=multi_delta1_color, linewidth=0, rounding_size=0.25)

axs[0,1].set_xticks(xticks)
axs[0,1].set_xticklabels(ops_depth, fontsize=10)
axs[0,1].set_xlabel("(b) Depth ($\delta$1"+r'$\uparrow$)', fontsize=12)
legend_patch = Patch(facecolor=single_delta1_color, edgecolor=single_delta1_color, label='Single-task')
legend_patch2 = Patch(facecolor=multi_delta1_color, edgecolor=multi_delta1_color, label='Multi-task')
axs[0,1].legend(handles=[legend_patch, legend_patch2], loc='best', fontsize=10)


ops_normal = ["NYUv2", "ScanNet", "iBims-1", "Sintel", ""]

mean_single = [21.8, 20.5, 23.7, 42.1]
mean_multi = [22.7, 21.1, 24.4, 42.9]
y_mean = [10.0, 50.0]
angle1125_single = [46.3, 46.5, 51.9, 10.9]
angle1125_multi = [43.0, 44.0, 49.0, 10.1]
y_angle1125 = [0.0, 80.0]

single_mean_color = 'lightsteelblue'
multi_mean_color = 'cornflowerblue'


axs[1,0].set_ylim(y_mean)
for xi, yi in zip(xticks[:5], mean_single):
    rounded_bar(axs[1,0], xi, yi, width=0.3, align="left", facecolor=single_mean_color, edgecolor=single_mean_color, linewidth=0, rounding_size=0.25)

for xi, yi in zip(xticks[:5], mean_multi):
    rounded_bar(axs[1,0], xi, yi, width=0.3, align="center", facecolor=multi_mean_color, edgecolor=multi_mean_color, linewidth=0, rounding_size=0.25)

axs[1,0].set_xticks(xticks)
axs[1,0].set_xticklabels(ops_normal, fontsize=10)
axs[1,0].set_xlabel("(c) Surface normal (Mean"+r'$\downarrow$)', fontsize=12)
legend_patch = Patch(facecolor=single_mean_color, edgecolor=single_mean_color, label='Single-task')
legend_patch2 = Patch(facecolor=multi_mean_color, edgecolor=multi_mean_color, label='Multi-task')
axs[1,0].legend(handles=[legend_patch, legend_patch2], loc='best', fontsize=10)

single_angle1125_color = 'powderblue'
multi_angle1125_color = 'deepskyblue'

axs[1,1].set_ylim(y_angle1125)
for xi, yi in zip(xticks[:5], angle1125_single):
    rounded_bar(axs[1,1], xi, yi, width=0.3, align="left", facecolor=single_angle1125_color, edgecolor=single_angle1125_color, linewidth=0, rounding_size=0.25)

for xi, yi in zip(xticks[:5], angle1125_multi):
    rounded_bar(axs[1,1], xi, yi, width=0.3, align="center", facecolor=multi_angle1125_color, edgecolor=multi_angle1125_color, linewidth=0, rounding_size=0.25)

axs[1,1].set_xticks(xticks)
axs[1,1].set_xticklabels(ops_normal, fontsize=10)
axs[1,1].set_xlabel("(d) Surface normal ($11.25^{\circ}$"+r'$\uparrow$)', fontsize=12)
legend_patch = Patch(facecolor=single_angle1125_color, edgecolor=single_angle1125_color, label='Single-task')
legend_patch2 = Patch(facecolor=multi_angle1125_color, edgecolor=multi_angle1125_color, label='Multi-task')
axs[1,1].legend(handles=[legend_patch, legend_patch2], loc='upper right', fontsize=10)

fig.tight_layout()
plt.show()


