# Add CUDA support

FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
# FROM ros:n-ros-core-bionic
ENV TZ=US \
    DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    # CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
# RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 lsb-core wget curl nano mercurial python3-pip\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
#     git \
#     build-essential \
#     cmake \
#     vim \
#     wget \
#     curl \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libglib2.0-0 \
#     lsb-core \
#     ninja-build \
#     pkg-config \
#     python-pip \
#     python3-pip \
#     python3-dev \
#     python-numpy \
#     libjpeg8-dev \
#     libtiff5-dev \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# Setup your sources list and keys
RUN apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    lsb-release \
    locales vim \
    && rm -rf /var/lib/apt/lists/*
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN dpkg-reconfigure locales

# Install mesa and opengl for gazebo
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Install ROS Noetic
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-noetic-desktop-full
RUN apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
RUN rosdep init \
 && rosdep fix-permissions \
 && rosdep update
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Elevation mapping specific dependencies
ENV FORCE_CUDA="1"

# 1: ROS deps, 2: Plane Seg deps 3: Catkin build deps
RUN apt install -y libopencv-dev libeigen3-dev libgmp-dev libmpfr-dev libboost-all-dev
RUN apt-get install -y ros-noetic-pybind11-catkin \
    ros-noetic-grid-map-core ros-noetic-grid-map-msgs ros-noetic-grid-map 
RUN apt-get install -y ros-noetic-catkin python3-catkin-tools


# Install elevation_mapping_cupy
# RUN conda clean --all
# COPY . /elevation_mapping_cupy
# WORKDIR /elevation_mapping_cupy

# CUDA python deps
RUN pip install cupy-cuda116 catkin_pkg 

# Traversability dependencies
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 chainer
# RUN pip install -r requirements.txt