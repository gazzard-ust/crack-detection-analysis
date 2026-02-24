"""Generate fig_architecture.png - YOLO-World XL architecture diagram."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.use('Agg')

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color, fontsize=8, subtext=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='#333333', linewidth=1.2, alpha=0.9)
    ax.add_patch(box)
    if subtext:
        ax.text(x + w/2, y + h/2 + 0.12, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white')
        ax.text(x + w/2, y + h/2 - 0.15, subtext, ha='center', va='center',
                fontsize=fontsize - 2, color='white', alpha=0.9)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white')

def draw_arrow(ax, x1, y1, x2, y2, color='#555555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Colors
c_input = '#607D8B'
c_backbone = '#1565C0'
c_neck = '#2E7D32'
c_text = '#6A1B9A'
c_head = '#D84315'
c_output = '#37474F'
c_repvl = '#F57C00'

# ---- Input ----
draw_box(ax, 0.2, 2.2, 1.4, 1.2, 'Input Image', c_input, 9, '640×640×3')

# ---- Text Encoder ----
draw_box(ax, 0.2, 4.5, 1.4, 1.0, 'CLIP Text\nEncoder', c_text, 8)
ax.text(0.9, 4.2, 'Class prompts:\n"dummy crack"\n"PVC pipe crack"\n"paper crack"', ha='center', va='top',
        fontsize=6, style='italic', color='#444')

# ---- Backbone (CSPDarknet) ----
bx = 2.4
draw_box(ax, bx, 3.6, 1.6, 0.8, 'Stage 3 (C3)', c_backbone, 8, 'P3: 80×80')
draw_box(ax, bx, 2.4, 1.6, 0.8, 'Stage 4 (C4)', c_backbone, 8, 'P4: 40×40')
draw_box(ax, bx, 1.2, 1.6, 0.8, 'Stage 5 (C5)', c_backbone, 8, 'P5: 20×20')
ax.text(bx + 0.8, 4.7, 'CSPDarknet-XL\nBackbone', ha='center', fontsize=9, fontweight='bold', color=c_backbone)

# Arrows input -> backbone
draw_arrow(ax, 1.6, 2.8, 2.4, 2.8)
draw_arrow(ax, 2.4, 2.8, 2.4, 4.0)
draw_arrow(ax, 2.4, 2.8, 2.4, 1.6)

# ---- Re-parameterizable VL-PAN (RepVL-PAN) ----
nx = 5.0
draw_box(ax, nx, 3.6, 1.8, 0.8, 'T-CSPLayer', c_repvl, 8, 'Text-guided')
draw_box(ax, nx, 2.4, 1.8, 0.8, 'T-CSPLayer', c_repvl, 8, 'Text-guided')
draw_box(ax, nx, 1.2, 1.8, 0.8, 'T-CSPLayer', c_repvl, 8, 'Text-guided')
ax.text(nx + 0.9, 4.7, 'RepVL-PAN\nNeck', ha='center', fontsize=9, fontweight='bold', color=c_repvl)

# I-PoolingAttn block
draw_box(ax, nx + 0.15, 0.2, 1.5, 0.7, 'Image Pooling\nAttention', '#795548', 7)

# Arrows backbone -> neck
draw_arrow(ax, 4.0, 4.0, 5.0, 4.0)
draw_arrow(ax, 4.0, 2.8, 5.0, 2.8)
draw_arrow(ax, 4.0, 1.6, 5.0, 1.6)

# Text encoder -> neck (text-guided fusion)
draw_arrow(ax, 1.6, 5.0, 5.0, 4.3, color=c_text)
draw_arrow(ax, 5.9, 0.9, 5.9, 1.2, color='#795548')

# FPN connections (top-down + bottom-up)
ax.annotate('', xy=(5.4, 3.6), xytext=(5.4, 3.2),
            arrowprops=dict(arrowstyle='->', color=c_repvl, lw=1.0, ls='--'))
ax.annotate('', xy=(6.4, 3.2), xytext=(6.4, 3.6),
            arrowprops=dict(arrowstyle='->', color=c_repvl, lw=1.0, ls='--'))
ax.annotate('', xy=(5.4, 2.4), xytext=(5.4, 2.0),
            arrowprops=dict(arrowstyle='->', color=c_repvl, lw=1.0, ls='--'))
ax.annotate('', xy=(6.4, 2.0), xytext=(6.4, 2.4),
            arrowprops=dict(arrowstyle='->', color=c_repvl, lw=1.0, ls='--'))

# ---- Contrastive Head ----
hx = 7.8
draw_box(ax, hx, 3.6, 2.0, 0.8, 'Contrastive\nHead (P3)', c_head, 8)
draw_box(ax, hx, 2.4, 2.0, 0.8, 'Contrastive\nHead (P4)', c_head, 8)
draw_box(ax, hx, 1.2, 2.0, 0.8, 'Contrastive\nHead (P5)', c_head, 8)
ax.text(hx + 1.0, 4.7, 'Detection\nHeads', ha='center', fontsize=9, fontweight='bold', color=c_head)

# Arrows neck -> head
draw_arrow(ax, 6.8, 4.0, 7.8, 4.0)
draw_arrow(ax, 6.8, 2.8, 7.8, 2.8)
draw_arrow(ax, 6.8, 1.6, 7.8, 1.6)

# Text embeddings -> heads
draw_arrow(ax, 1.6, 5.3, 8.8, 4.7, color=c_text)
ax.text(5.2, 5.3, 'Text embeddings', fontsize=7, color=c_text, style='italic', ha='center')

# ---- NMS + Output ----
ox = 10.6
draw_box(ax, ox, 2.4, 1.5, 0.8, 'Region-Text\nMatching', '#455A64', 8)
draw_arrow(ax, 9.8, 4.0, 10.6, 3.0)
draw_arrow(ax, 9.8, 2.8, 10.6, 2.8)
draw_arrow(ax, 9.8, 1.6, 10.6, 2.6)

draw_box(ax, 12.5, 2.4, 1.3, 0.8, 'Predictions', c_output, 8, 'BBox + Class')
draw_arrow(ax, 12.1, 2.8, 12.5, 2.8)

# ---- Title ----
ax.text(7.0, 5.8, 'YOLO-World XL Architecture for Pipe Crack Detection', ha='center',
        fontsize=13, fontweight='bold')

# ---- Legend ----
legend_items = [
    ('Image Backbone', c_backbone),
    ('Text Encoder', c_text),
    ('RepVL-PAN Neck', c_repvl),
    ('Detection Heads', c_head),
]
for i, (label, color) in enumerate(legend_items):
    ax.add_patch(FancyBboxPatch((0.3 + i * 2.8, 0.15), 0.3, 0.3,
                                boxstyle="round,pad=0.05", facecolor=color, edgecolor='#333'))
    ax.text(0.75 + i * 2.8, 0.3, label, fontsize=7, va='center')

plt.savefig('fig_architecture.png', bbox_inches='tight', facecolor='white')
print("Saved fig_architecture.png")
