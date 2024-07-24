#!/bin/bash -e

docker_image=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

docker run -u $(id -u ${USER}):$(id -g ${USER}) \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v $(pwd):/workspace \
    --gpus all \
    --rm \
    -it \
    ${docker_image} /bin/bash
