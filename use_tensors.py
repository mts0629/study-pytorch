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
    array_from_x = x.numpy()

    assert torch.equal(x, x_from_arr)
    assert np.all(array == array_from_x)

    # Autograd
    print("# Autograd")
    w = torch.tensor(
        [[0.1, 0.2],
         [0.3, 0.4]],
        dtype=torch.float32,
        requires_grad=True
    )
    b = torch.ones(2, requires_grad=True)

    x.requires_grad_()  # x.requires_grad = True

    y = torch.matmul(x, w) + b
    t = torch.zeros([2, 2])
    print(y)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(y, t)

    loss.backward()  # Backward need to start from scalar output
    print("Grads:")
    print(x.grad)
    print(w.grad)
    print(b.grad)

    with torch.no_grad():  # Disable gradient tracking
        y = torch.matmul(x, w) + b

    assert y.requires_grad == False

    print("# Quantize")
    x = torch.rand([5], dtype=torch.float32)
    print(x)
    print("# Per-Tensor")
    q_x = torch.quantize_per_tensor(
        x, scale=0.1, zero_point=-128, dtype=torch.qint8
    )
    print(q_x)
    print("=", q_x.int_repr())

    print("# Per-Channel")
    q_x = torch.quantize_per_channel(
        x,
        scales=torch.Tensor([0.1, 0.125, 0.15, 0.175, 0.2]),
        zero_points=torch.Tensor([-128, -64, -32, -16, -8]),
        axis=0,
        dtype=torch.qint8
    )
    print(q_x)
    print("=", q_x.int_repr())
