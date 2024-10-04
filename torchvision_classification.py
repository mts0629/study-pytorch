"""Use pre-trained models via Torchvision."""

import torch.hub as hub
import torchvision.models as models
from torchvision.io import read_image

if __name__ == "__main__":
    hub.set_dir("work")  # Cache directory for weights

    # MobileNet V2 with pretrained weights
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    print(model)

    # Get an appropriate preprocess from initial weights
    preprocess = weights.transforms()

    model.eval()

    # Run an inference
    image = read_image("work/data/dog.jpg")
    x = preprocess(image).unsqueeze(0)
    y = model(x).squeeze(0).softmax(0)

    # Output label
    label = y.argmax(0).item()
    print("##########")
    print(f"{weights.meta['categories'][label]} ({label}, y={y.max()})")
