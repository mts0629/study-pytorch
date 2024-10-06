"""Use pre-trained models for Instance Segmentation via Torchvision."""

import torch.hub as hub
import torchvision.models.detection as detection

from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks

if __name__ == "__main__":
    hub.set_dir("work")

    # Mask R-CNN
    weights = detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = detection.maskrcnn_resnet50_fpn(weights=weights)
    # print(model)

    preprocess = weights.transforms()

    model.eval()

    image = read_image("work/data/town.jpg")
    x = preprocess(image).unsqueeze(0)
    y = model(x)[0]  # List[Dict]
                     # Keys: ['boxes', 'labels', 'scores', 'masks']

    # Remove low-score candidates
    score_threshold = 0.5
    indices = [i for i, y in enumerate(y["scores"]) if y > score_threshold]
    masks = y["masks"][indices]

    # Get boolean masks by probability threshold = 0.5
    prob_threshold = 0.5
    bool_masks = (masks > prob_threshold)\
        .squeeze(1)  # Remove an extra dimension

    # Print labels and scores of detection results
    labels = [
        weights.meta["categories"][label] for label in y["labels"][indices]
    ]
    print(f"{len(labels)} objects:")
    print([
        f"{label} ({score})"
        for label, score in zip(labels, y["scores"][indices])
    ])

    # Draw segmentation masks
    result = draw_segmentation_masks(image, bool_masks, alpha=0.8).detach()

    to_pil_image(result).save("work/data/mask_r-cnn_segmentation.png")
