

import json
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
import numpy as np

# Option to overlay masks
overlay_mask = True  # Set to True to overlay segmentation masks
# Option to plot bounding boxes
plot_bbox = True    # Set to True to plot bounding boxes
plot_outline = True  # Set to True to plot polygon outlines

# Paths
#json_path = r'./TRAINDATA/annotations/instances_val2017.json'
#images_dir = r'./TRAINDATA/val2017'
#image_filename = '7c591d6d-26bc-4ae2-b.tif' 

json_path = r'./TRAINDATA/sliced_coco/annotations/instances_val2017.json'
images_dir = r'./TRAINDATA/sliced_coco/val2017'
image_filename = '0492a795-9453-4099-9_0_960_640_1360_1040.png' 

# Load COCO JSON
with open(json_path, 'r') as f:
    coco = json.load(f)

# Find image info
image_info = next(img for img in coco['images'] if img['file_name'] == image_filename)
image_id = image_info['id']

# Find annotations for this image
anns = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]

# Load image
img_path = os.path.join(images_dir, image_filename)
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Prepare mask overlay
if overlay_mask:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for ann in anns:
        if 'segmentation' in ann and isinstance(ann['segmentation'], list):
            for seg in ann['segmentation']:
                pts = np.array([(seg[i], seg[i+1]) for i in range(0, len(seg), 2)], np.int32)
                cv2.fillPoly(mask, [pts], 1)
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = [0, 255, 255]  # Cyan mask
    # Blend image and mask
    alpha = 0.4
    image = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)

# Plot image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)

# Plot each annotation (polygon and bbox)
for ann in anns:
    if 'segmentation' in ann and isinstance(ann['segmentation'], list):
        if plot_outline: 
            for seg in ann['segmentation']:
                poly = Polygon(
                    [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)],
                    closed=True, fill=False, edgecolor='c', linewidth=1.2
                )
                ax.add_patch(poly)
    if plot_bbox and 'bbox' in ann:
        x, y, w, h = ann['bbox']
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='c', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
#plt.show()
outfile = image_filename.replace('.tif', '_coco_gt.png')
plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=600)
print(f"Saved plot to {outfile}")