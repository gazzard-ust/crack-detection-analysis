"""Generate fig_samples.png - Annotated sample images (one per class)."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

BASE = 'pipe-crack-detection-1/test'
CLASS_NAMES = {0: 'Dummy Crack', 1: 'PVC Pipe Crack', 2: 'Paper Crack'}
CLASS_COLORS = {0: (66, 133, 244), 1: (234, 67, 53), 2: (52, 168, 83)}  # BGR

# Order: PVC Pipe Crack, Paper Crack, Dummy Crack
samples = [
    (1, 'original-2025-12-20T131033-510_jpg.rf.3df9f8f1a41807715169a88bc376422a'),
    (2, 'IMG_20251213_191012_jpg.rf.92466bc2c04122800cd5657139df2e38'),
    (0, 'IMG_7994_jpeg.rf.f91d8746d93e88aba3d7984573b83a14'),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for idx, (cls_id, stem) in enumerate(samples):
    img_path = os.path.join(BASE, 'images', stem + '.jpg')
    lbl_path = os.path.join(BASE, 'labels', stem + '.txt')

    img = cv2.imread(img_path)
    if img is None:
        print(f"WARNING: Could not read {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Parse YOLO annotations and draw bounding boxes from polygon/bbox
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            c = int(parts[0])
            coords = list(map(float, parts[1:]))

            if len(coords) == 4:
                # Standard YOLO bbox: cx, cy, w, h
                cx, cy, bw, bh = coords
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
            else:
                # Polygon format: x1,y1,x2,y2,...
                xs = [coords[i] * w for i in range(0, len(coords), 2)]
                ys = [coords[i] * h for i in range(1, len(coords), 2)]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))

                # Also draw the polygon
                pts = np.array([[int(coords[i]*w), int(coords[i+1]*h)]
                                for i in range(0, len(coords), 2)], np.int32)
                cv2.polylines(img, [pts], True, CLASS_COLORS[c][::-1], 2)

            color_rgb = CLASS_COLORS[c][::-1]
            cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb, 2)

            label = CLASS_NAMES[c]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_rgb, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    ax = axes[idx]
    ax.imshow(img)
    ax.set_title(f'({chr(97+idx)}) {CLASS_NAMES[cls_id]}', fontsize=11, fontweight='bold')
    ax.axis('off')

fig.suptitle('Annotated Sample Images from Each Crack Class', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_samples.png', bbox_inches='tight', facecolor='white')
print("Saved fig_samples.png")
