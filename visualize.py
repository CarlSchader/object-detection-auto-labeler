# takes an image url or file path and bounding boxes
# and displays the image with the boxes

import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize image with bounding boxes')
    parser.add_argument('image', type=str, help='Image url or file path')
    parser.add_argument('boxes', type=str, help='Bounding boxes')
    parser.add_argument('--labels', type=str, help='Labels for bounding boxes', default=None)
    args = parser.parse_args()

    boxes = np.array([[float(val) for val in box.split(',')] for box in args.boxes.split(':')])
    if args.labels is not None:
        labels = args.labels.split(':')
    else:
        labels = [str(i) for i in range(len(boxes))]

    visualize(args.image, boxes, labels)