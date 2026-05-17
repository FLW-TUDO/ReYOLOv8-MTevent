"""
Generate updated IEEE-style ablation bar chart with full GEN1 clip-length results.
Output: ablation_figure.pdf + .png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
labels   = ['YOLO\nC1',
            'ReY\nC3', 'ReY\nC7', 'ReY\nC11', 'ReY\nC21',
            'GEN1\nC3', 'GEN1\nC7', 'GEN1\nC11', 'GEN1\nC21',
            'PEDRo\nC11', 'PEDRo\nC21']

map50    = [0.260,
            0.262, 0.271, 0.261, 0.285,
            0.293, 0.324, 0.324, 0.329,
            0.251, 0.258]

map9595  = [0.138,
            0.148, 0.152, 0.141, 0.155,
            0.157, 0.181, 0.178, 0.164,
            0.134, 0.138]

# Group boundaries (index of first bar in each group)
groups = [
    (0,  1,  'Non-rec.\n(scratch)'),
    (1,  5,  'Recurrent\n(scratch)'),
    (5,  9,  'GEN1\n(pretrained)'),
    (9, 11,  'PEDRo\n(pretrained)'),
]

n = len(labels)
x = np.arange(n)
w = 0.38

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 2.9))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── Bars ──────────────────────────────────────────────────────────────────────
COLOR_50   = '#2c3e7a'   # dark blue solid
COLOR_9595 = '#8a9cc8'   # light blue hatched

bars50   = ax.bar(x - w/2, map50,   width=w, color=COLOR_50,   label='mAP50',    zorder=3)
bars9595 = ax.bar(x + w/2, map9595, width=w, color=COLOR_9595,
                  hatch='//', edgecolor='white', label='mAP50-95', zorder=3)

# Highlight GEN1 C21 best bar with black edge
bars50[9-1].set_edgecolor('black')   # GEN1 C21 is index 8 (0-based)
bars50[8].set_linewidth(1.5)

# Value labels on mAP50 bars
for bar, val in zip(bars50, map50):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=4.2, color='black')

# ── Group separator lines and top labels ──────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=5.5)

y_top    = 0.36
y_label  = 0.355
sep_y    = 0.005   # where vertical separators sit (data coords)

for i, (start, end, gname) in enumerate(groups):
    mid   = (start + end - 1) / 2
    left  = start - 0.5
    right = end   - 0.5

    # Dashed vertical separator before each group (except first)
    if start > 0:
        ax.axvline(x=left, color='gray', linestyle='--', linewidth=0.6,
                   ymin=0, ymax=0.93, zorder=1)

    # Horizontal bracket line above bars
    ax.annotate('', xy=(right - 0.05, y_top), xytext=(left + 0.05, y_top),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.7),
                annotation_clip=False)

    # Group name
    ax.text(mid, y_label + 0.007, gname, ha='center', va='bottom',
            fontsize=4.5, color='#333333', fontweight='bold',
            transform=ax.transData, clip_on=False)

# ── Axes styling ──────────────────────────────────────────────────────────────
ax.set_ylabel('mAP', fontsize=7)
ax.set_ylim(0, 0.395)
ax.set_xlim(-0.6, n - 0.4)
ax.yaxis.grid(True, linestyle='--', linewidth=0.4, color='#cccccc', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=6)
ax.tick_params(axis='x', length=0)

# ── Legend below axes ─────────────────────────────────────────────────────────
patch50   = mpatches.Patch(color=COLOR_50,   label='mAP50')
patch9595 = mpatches.Patch(facecolor=COLOR_9595, hatch='//',
                            edgecolor='white', label='mAP50-95')
ax.legend(handles=[patch50, patch9595], loc='upper center',
          bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=6,
          frameon=False)

plt.tight_layout(rect=[0, 0.04, 1, 1])

# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = '/tmp/paper_latex/figures/comparison_figures/ablation_figure.pdf'
out_png = '/home/loki/event/paper/ablation_figure_v2.png'

plt.savefig(out_pdf, bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig(out_png, bbox_inches='tight', dpi=300, facecolor='white')
print(f"Saved:\n  {out_pdf}\n  {out_png}")
