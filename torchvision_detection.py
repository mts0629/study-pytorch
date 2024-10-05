"""Use pre-trained models for Object Detection via Torchvision."""

# import torch
import torch.hub as hub
import torchvision.models.detection as detection

from torchvision.io import read_image
from torchvision.ops import nms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

if __name__ == "__main__":
    hub.set_dir("work")

    # SSD 300
    weights = detection.SSD300_VGG16_Weights.COCO_V1
    model = detection.ssd300_vgg16(weights=weights)
    # print(model)

    preprocess = weights.transforms()

    model.eval()

    image = read_image("work/data/horses.jpg")
    x = preprocess(image).unsqueeze(0)
    y = model(x)[0]  # List[Dict]
                     # Keys: ['boxes', 'scores', 'labels']

    # Extract candidates which score > 0.25
    score_threshold = 0.25
    score_indices = [
        i for i, y in enumerate(y["scores"]) if y > score_threshold
    ]

    # Remove candidates by NMS with IoU threshold = 0.5
    bbox_indices = nms(
        boxes=y["boxes"][score_indices, :],
        scores=y["scores"][score_indices],
        iou_threshold=0.5
    )
    bboxes = y["boxes"][bbox_indices, :]
    labels = [
        weights.meta["categories"][i] for i in y["labels"][bbox_indices]
    ]

    # Print labels and scores of detection results
    print([
        f"{label} ({score})"
        for label, score in zip(labels, y["scores"][bbox_indices])
    ])

    # Draw bounding boxes
    result = draw_bounding_boxes(
        image, bboxes, labels=labels, width=16
    ).detach()  # Copy a tensor

    to_pil_image(result).save("work/data/ssd300_detection.png")
