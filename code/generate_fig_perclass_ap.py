"""Generate fig_perclass_ap.png - Bar chart of per-class AP@50-95."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

classes = ['Dummy Crack', 'PVC Pipe Crack', 'Paper Crack']
ap50_95 = [95.60, 91.16, 74.00]
ap50 = [99.50, 98.61, 97.12]

x = np.arange(len(classes))
width = 0.32

fig, ax = plt.subplots(figsize=(7, 4.5))

bars1 = ax.bar(x - width/2, ap50, width, label='AP@50', color='#2196F3', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, ap50_95, width, label='AP@50-95', color='#FF9800', edgecolor='white', linewidth=0.5)

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Average Precision (%)')
ax.set_title('Per-Class Detection Performance on Test Set', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('fig_perclass_ap.png', bbox_inches='tight', facecolor='white')
print("Saved fig_perclass_ap.png")
