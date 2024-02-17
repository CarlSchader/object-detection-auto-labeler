# takes an image url or file path and bounding boxes
# and displays the image with the boxes

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

import argparse

from utils import load_im

def visualize(image_path, boxes, labels, confidences, threshold=-1.0):
    img = load_im(image_path)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)

    for box, label, confidence in zip(boxes, labels, confidences):
        if confidence < threshold:
            continue

        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, label, color='white', fontsize=12, ha='left', va='bottom')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize image with bounding boxes')
    parser.add_argument('image', type=str, help='Image url or file path')
    parser.add_argument('--boxes', '-b', type=str, help='Bounding boxes', default=None)
    parser.add_argument('--labels', '-l', type=str, help='Labels for bounding boxes', default=None)
    parser.add_argument('--confidences', '-c', type=str, help='Confidences for bounding boxes', default=None)
    parser.add_argument('--input_file', '-f', type=str, help='Input file path', default=None)
    parser.add_argument('--threshold', '-t', type=float, help='Confidence threshold for showing bounding boxes', default=-1.0)
    args = parser.parse_args()

    # defaults
    boxes = []
    labels = [str(i) for i in range(len(boxes))]
    confidences = [1.0 for i in range(len(boxes))]

    # populate the boxes and labels arrays
    if args.input_file is not None:
        data = json.load(open(args.input_file, "r"))
        boxes = [box['location'] for box in data]
        labels = [box['label'] for box in data]
        confidences = [box['confidence'] for box in data]

    if args.boxes is not None:
        boxes = [[[float(val) for val in box.split(',')] for box in args.boxes.split(':')]]
    
    if args.labels is not None:
        labels = args.labels.split(':')
    
    if args.confidences is not None:
        confidences = [float(conf) for conf in args.confidences.split(':')]

    visualize(args.image, boxes, labels, confidences, args.threshold)