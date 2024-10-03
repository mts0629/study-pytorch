"""Use pre-trained models via Torchvision."""

import torchvision.models as models
from torchvision.io import read_image


if __name__ == "__main__":
    # MobileNet V2 with pretrained weights
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )
    # OR
    # model = models.get_model("mobilenet_v2", weights="DEFAULT")

    print(model)

    # Get an appropriate preprocess from initial weights
    preprocess = models.MobileNet_V2_Weights.DEFAULT.transforms()

    model.eval()

    # Run an inference and show the result (class id)
    image = read_image("test/image.png")
    x = preprocess(image).unsqueeze(0)
    y = model(x).squeeze(0).softmax(0)
    print(y.argmax(0).item())
