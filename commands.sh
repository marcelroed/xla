# Command to launch container runtime
srun --time=240 -A nvr_lacr_llm --container-image=tensorflow/build:latest-python3.11 --container-mount-home --container-mounts=$PWD:/xla,/home/yusu/new_home:/home/yusu/new_home/ --container-workdir=/xla --partition interactive --gpus-per-node=8 --cpus-per-gpu=28 --nodes=1 --pty bash

# For running
srun --time=240 -A nvr_lacr_llm --container-image=docker://nvcr.io#nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 --container-mount-home --container-mounts=/home/yusu/new_home:/home/yusu/new_home/ --container-workdir=/home/yusu/new_home/ --partition interactive --gpus-per-node=8 --cpus-per-gpu=28 --nodes=1 --pty bash

# Compiling XLA
srun --time=240 -A nvr_lacr_llm --container-image=docker://nvcr.io#nvidia/cuda-dl-base:25.02-cuda12.8-devel-ubuntu24.04 --container-mount-home --container-mounts=/home/yusu/new_home:/home/yusu/new_home/ --container-workdir=/home/yusu/new_home/ --partition interactive --gpus-per-node=8 -c 224 --gpus 8 --pty bash


# Install clang-17
apt update
apt install clang-17

# Configure correctly
uv run -p 3.11 --no-project ./configure.py --backend=CUDA --cuda_version 12.8.0 --cudnn_version 9.7.1 --nccl --clang_path=/usr/bin/clang-17

# Build with container support in the JVM
pixi x bazel --host_jvm_args=-XX:-UseContainerSupport build --linkopt=-lm --test_output=all //xla/...


# Build Jax
# uv run -p 3.11 --no-project python build/build.py build --wheels=jaxlib,jax-cuda-pjrt,jax-cuda-plugin --python_version=3.11 --local_xla_path=../xla --cuda_version=12.6.2 --cudnn_version=9.7.1 --clang_path /usr/bin/clang-17 --bazel_startup_options="--host_jvm_args=-XX:-UseContainerSupport"
pixi x -s clang==18* -s libstdcxx-ng -s libstdcxx-devel_linux-64 uv run -p 3.11 --no-project python build/build.py build --wheels=jaxlib,jax-cuda-pjrt,jax-cuda-plugin --python_version=3.11 --local_xla_path=../xla --cuda_version=12.6.2 --cudnn_version=9.7.1 --bazel_startup_options="--host_jvm_args=-XX:-UseContainerSupport"