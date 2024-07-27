# study-pytorch

Study of PyTorch

- [Pytorch](https://pytorch.org)
    - [Docs](https://pytorch.org/docs/stable/index.html)
    - [Tutorials](https://pytorch.org/tutorials)

## Environment

- NVIDIA GeForce RTX 2060 (6GB VRAM)
- Windows 10
- WSL2
    - Distro: Ubuntu 18.04.6 LTS
- PyTorch Docker image (CUDA support)
    - [pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime](https://hub.docker.com/layers/pytorch/pytorch/2.3.0-cuda12.1-cudnn8-runtime/images/sha256-0279f7aa29974bf64e61d0ff6e979b41a249b3662a46e30778dbf80b8c99c361?context=explore)

`Dockerfile` and a runner script: `docker_run.sh` are in `docker/`.

This script uses a host user for login.

```sh
$ ./docker/docker_run.sh <IMAGE_TAG>
```

Also the VSCode Dev Container can be used.

## Prerequisites

- NVIDIA GPU Drivers
- Docker
- NVIDIA container ToolKit
    - [Installing the NVIDIA Container ToolKit](
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)
- Visual Studio Code + Dev Containers extension (for Dev Container)
