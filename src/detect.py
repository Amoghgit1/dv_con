import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F
from PIL import Image
import os

COCO_LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

model = None

def load_model():
    global model
    if model is None:
        print("Loading MobileNet-SSD model...")
        model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
        model.eval()
        print("Model loaded.")

def detect_objects(image_path, confidence_threshold=0.4):
    load_model()
    image = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(tensor)[0]

    results = []
    for label_idx, score, box in zip(
        predictions['labels'], predictions['scores'], predictions['boxes']
    ):
        if score.item() < confidence_threshold:
            continue
        label = COCO_LABELS[label_idx.item()]
        if label == 'N/A':
            continue
        results.append({
            'label': label,
            'confidence': round(score.item(), 3),
            'box': [round(x, 1) for x in box.tolist()]
        })

    return results

if __name__ == "__main__":
    test_image = "../test_images/kitchen.jpg"
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        print("Please add images to the test_images/ folder first.")
    else:
        detections = detect_objects(test_image)
        print(f"Detected {len(detections)} objects:")
        for d in detections:
            print(f"  {d['label']} ({d['confidence']})")