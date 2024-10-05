"""Use pre-trained models for Semantic Segmentation via Torchvision."""

import torch
import torch.hub as hub
import torchvision.models as models
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

if __name__ == "__main__":
    hub.set_dir("work")

    # DeepLab V3
    weights = models.get_weight(
        "DeepLabV3_ResNet50_Weights.DEFAULT"
    )
    model = models.get_model(
        "deepLabv3_resnet50", weights=weights
    )
    # print(model)

    preprocess = weights.transforms()

    model.eval()

    image = read_image("work/data/town.jpg")
    x = preprocess(image).unsqueeze(0)
    y = model(x)  # Output is OrderdDict: ['out', 'aux']

    # Get confidences for each class
    normalized_masks = y["out"].softmax(dim=1)

    # Classes:
    # '__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
    # 'bottle', 'bus', 'car', 'cat', 'chair',
    # 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    # 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    class_to_index = {
        cls: idx for (idx, cls) in enumerate(weights.meta["categories"])
    }

    # Get mask of "car" and "person" and then concatenate them as an RGB tensor
    mask_car = normalized_masks[0, class_to_index["car"]]
    mask_person = normalized_masks[0, class_to_index["person"]]
    output = torch.stack(
        (
            (mask_car * 255).to(torch.uint8),  # R: "car"
            (mask_person * 255).to(torch.uint8),  # G: "person"
            torch.zeros(normalized_masks[0, 0].shape, dtype=torch.uint8)  # B: 0
        ),
        dim=0
    )

    # Output the RGB tensor as an image
    to_pil_image(output).save("work/data/deeplabv3_segmentation.jpg")
