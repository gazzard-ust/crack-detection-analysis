"""Generate fig_training_curves.png - Training/validation loss and mAP curves over 100 epochs."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

df = pd.read_csv('crack_detection_runs/yoloworld_xl_20251226_171626/results.csv')
df.columns = df.columns.str.strip()
epochs = df['epoch'] + 1  # 1-indexed

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# (a) Box Loss
ax = axes[0, 0]
ax.plot(epochs, df['train/box_loss'], label='Train', color='#2196F3', linewidth=1.2)
ax.plot(epochs, df['val/box_loss'], label='Val', color='#F44336', linewidth=1.2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Box Loss')
ax.set_title('(a) Box Loss')
ax.legend(framealpha=0.8)

# (b) Classification Loss
ax = axes[0, 1]
ax.plot(epochs, df['train/cls_loss'], label='Train', color='#2196F3', linewidth=1.2)
ax.plot(epochs, df['val/cls_loss'], label='Val', color='#F44336', linewidth=1.2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Classification Loss')
ax.set_title('(b) Classification Loss')
ax.legend(framealpha=0.8)

# (c) DFL Loss
ax = axes[1, 0]
ax.plot(epochs, df['train/dfl_loss'], label='Train', color='#2196F3', linewidth=1.2)
ax.plot(epochs, df['val/dfl_loss'], label='Val', color='#F44336', linewidth=1.2)
ax.set_xlabel('Epoch')
ax.set_ylabel('DFL Loss')
ax.set_title('(c) Distribution Focal Loss')
ax.legend(framealpha=0.8)

# (d) mAP curves
ax = axes[1, 1]
ax.plot(epochs, df['metrics/mAP50(B)'], label='mAP@50', color='#4CAF50', linewidth=1.2)
ax.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@50-95', color='#FF9800', linewidth=1.2)
ax.plot(epochs, df['metrics/precision(B)'], label='Precision', color='#9C27B0', linewidth=1.0, linestyle='--', alpha=0.7)
ax.plot(epochs, df['metrics/recall(B)'], label='Recall', color='#00BCD4', linewidth=1.0, linestyle='--', alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Score')
ax.set_title('(d) Validation Metrics')
ax.legend(framealpha=0.8, fontsize=8)
ax.set_ylim(0.6, 1.0)

plt.tight_layout()
fig.suptitle('Training and Validation Metrics over 100 Epochs', fontsize=13, fontweight='bold', y=1.04)
plt.savefig('fig_training_curves.png', bbox_inches='tight', facecolor='white')
print("Saved fig_training_curves.png")
