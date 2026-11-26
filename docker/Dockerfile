ARG PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CXX=g++ \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    TRITON_CACHE_DIR=/tmp/triton \
    CUDA_CACHE_PATH=/tmp/cuda

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        build-essential git ninja-build \
        libboost-iostreams-dev libboost-system-dev libboost-thread-dev \
        libopencv-dev pybind11-dev python3-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# MiniZero's CMake also requires ALE for the Go target. Use its upstream pin.
# The pinned ALE headers rely on transitive <cstdint> includes that GCC 13 removed.
RUN git init --quiet /tmp/minizero-ale \
    && git -C /tmp/minizero-ale fetch --quiet --depth 1 \
        https://github.com/Farama-Foundation/Arcade-Learning-Environment.git \
        d59d00688b58c5c14dff5fc79db5c22e86987f5d \
    && git -C /tmp/minizero-ale checkout --quiet --detach FETCH_HEAD \
    && cmake -S /tmp/minizero-ale -B /tmp/minizero-ale/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_LIB=OFF \
        -DCMAKE_CXX_FLAGS="-include cstdint" \
    && cmake --build /tmp/minizero-ale/build --parallel 2 \
    && cmake --install /tmp/minizero-ale/build \
    && rm -rf /tmp/minizero-ale

ENV PG_MAJOR=19

# Keep the PGDG cluster configuration in the image and initialize data on the volume.
RUN . /etc/os-release \
    && install -d /etc/apt/keyrings \
    && wget --no-verbose --timeout=30 --tries=3 -O /etc/apt/keyrings/postgresql.asc \
        https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    && fingerprint=$(gpg --batch --show-keys --with-colons /etc/apt/keyrings/postgresql.asc | awk -F: '$1 == "fpr" { print $10; exit }') \
    && test "$fingerprint" = B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8 \
    && printf 'deb [signed-by=/etc/apt/keyrings/postgresql.asc] https://apt.postgresql.org/pub/repos/apt %s-pgdg main\n' \
        "$VERSION_CODENAME" > /etc/apt/sources.list.d/postgresql.list \
    && apt-get -o Acquire::Retries=3 -o Acquire::Languages=none update \
    && DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 install --yes --no-install-recommends \
        "postgresql-$PG_MAJOR" python3-psycopg \
    && python -c 'import psycopg' \
    && pg_conftool "$PG_MAJOR" main set listen_addresses localhost \
    && rm -rf "/var/lib/postgresql/$PG_MAJOR/main" /var/lib/apt/lists/*

# Reuse the base Python packages, including PyTorch, Psycopg, and pip.
RUN python -m venv --system-site-packages --without-pip /opt/xiaqi-venv

ENV VIRTUAL_ENV=/opt/xiaqi-venv \
    PATH="/opt/xiaqi-venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir tensorboard==2.21.0 \
    && python -c 'import psycopg; from filelock import FileLock; from torch.utils.tensorboard import SummaryWriter' \
    && tensorboard --version

WORKDIR /workspace
ENTRYPOINT []
CMD ["/bin/bash"]
