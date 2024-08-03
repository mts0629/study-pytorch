FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Timezone setting
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && apt-get upgrade -y && \
    apt-get install -y tzdata
ENV TZ Asia/Tokyo

# Install packages
RUN apt-get install -y git vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*
RUN pip install matplotlib

# Add a non-root user (overwrite by Dev Container)
ARG USER=dev
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID dev \
    && useradd -m -u $UID -g $GID dev

# Create a new home directory (used while the startup of Dev Container)
ARG HOME_DIR=/home/dev
RUN mkdir -p $HOME_DIR
RUN chown -R $UID:$GID $HOME_DIR

USER $USER
ENV HOME $HOME_DIR

WORKDIR /workspace
