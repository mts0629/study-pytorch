"""Use PyTorch Tensors."""

import numpy as np
import torch


if __name__ == "__main__":
    # Initialize
    x = torch.tensor(
        [[1, 2,],
         [3, 4]],
        dtype=torch.float32  # 32-bit float
    )
    print(x)
    print(f"{x.shape}, {x.dtype}, {x.device}, {x.layout}")

    # Conversion between NumPy ndarray
    array = np.array(
        [[1, 2],
         [3, 4]],
        dtype=np.float32
    )
    x_from_arr = torch.from_numpy(array)
    print(x_from_arr)

    array_from_x = x.numpy()
    print(array_from_x)

    # Autograd
    print("# Autograd")
    y = torch.tensor(
        [[5, 6],
         [7, 8]],
        dtype=torch.float32,
        requires_grad=True
    )

    x.requires_grad_()  # x.requires_grad = True

    z = x * y
    z = z.sum()  # Need to create scalar outputs
    z.backward()
    print(x.grad)
    print(y.grad)

    with torch.no_grad():
        z = x * y
    print(z.requires_grad)
