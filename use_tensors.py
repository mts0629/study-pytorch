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
