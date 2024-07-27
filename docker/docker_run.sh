#!/bin/bash -eu

docker_image=$1

# Login as the same user with the host
docker run -u $(id -u ${USER}):$(id -g ${USER}) \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v $(pwd):/workspace \
    --gpus all \
    --rm \
    -it \
    ${docker_image} /bin/bash
