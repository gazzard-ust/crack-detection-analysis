"""Generate fig_qualitative.png - Detection results with predicted bboxes and confidence scores.
Layout: 2x3 grid where each column is the same class.
Column order: PVC Pipe Crack, Paper Crack, Dummy Crack (matching fig_samples order).
"""
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

CLASS_NAMES = {0: 'Dummy Crack', 1: 'PVC Pipe Crack', 2: 'Paper Crack'}
CLASS_COLORS_RGB = {0: (66, 133, 244), 1: (234, 67, 53), 2: (52, 168, 83)}

# Columns: PVC Pipe Crack (1), Paper Crack (2), Dummy Crack (0)
# 2 images per class (top row, bottom row), different from fig_samples
column_order = [1, 2, 0]
column_images = {
    1: [  # PVC Pipe Crack - 2 isolated images
        'IMG_9800_jpeg.rf.8cbb35872024091c50b527b4297868f6',
        'original-28-_jpg.rf.1501f3ed36a98e3a07d85c07a69e57f6',
    ],
    2: [  # Paper Crack - 2 isolated images
        'IMG_1253_jpg.rf.2fb5b124f854290074a4517ee91d4cd7',
        'IMG_7854_jpeg.rf.e3ed482d90c0eeba32e210bbff9d1593',
    ],
    0: [  # Dummy Crack - 2 isolated images
        'IMG_7972_jpeg.rf.3fa964a97cc8f2aec076151a12bb76bc',
        'IMG_8002_jpeg.rf.3b345d0d5adb7e626d635c6bdf4f212d',
    ],
}

try:
    from ultralytics import YOLO
    model = YOLO('crack_detection_runs/yoloworld_xl_20251226_171626/weights/best.pt')

    test_dir = 'pipe-crack-detection-1/test/images'

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for col, cls_id in enumerate(column_order):
        for row, stem in enumerate(column_images[cls_id]):
            img_path = os.path.join(test_dir, stem + '.jpg')
            results = model.predict(img_path, conf=0.25, verbose=False)
            result = results[0]

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0].cpu().item())

                color = CLASS_COLORS_RGB.get(cls, (128, 128, 128))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                label = f'{CLASS_NAMES.get(cls, f"cls{cls}")} {conf:.2f}'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(img, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)

            ax = axes[row, col]
            ax.imshow(img)
            ax.axis('off')

    # Add column titles
    for col, cls_id in enumerate(column_order):
        axes[0, col].set_title(CLASS_NAMES[cls_id], fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig.suptitle('Qualitative Detection Results on Test Images', fontsize=14, fontweight='bold', y=1.015)
    plt.savefig('fig_qualitative.png', bbox_inches='tight', facecolor='white')
    print("Saved fig_qualitative.png (model inference)")

except Exception as e:
    print(f"Model inference failed: {e}")
    import traceback
    traceback.print_exc()
