# takes an image url or file path and bounding boxes
# and displays the image with the boxes

import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize(image, boxes, labels):
    # If image is a url
    if image.startswith('http'):
        response = requests.get(image)
        img = Image.open(BytesIO(response.content))
        img = np.array(img)
    else:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)

    for box, label in zip(boxes, labels):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, label, color='white', fontsize=12, ha='left', va='bottom')

    plt.show()