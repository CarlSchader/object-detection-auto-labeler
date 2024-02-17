import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

import argparse
from utils import load_im

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='infer bounding boxes and classes based on image and text array')
    parser.add_argument('image_location', type=str, help='Image url or file path')
    parser.add_argument('texts', type=str, help='Text array')
    args = parser.parse_args()

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    image = load_im(args.image_location)
    texts = [args.texts.split(':')]

    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Print detected objects and rescaled box coordinates
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
