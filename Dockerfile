FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# not apt update: most effective code in pytorch base image is in /opt/conda

WORKDIR /app

# Install pip dependencies

COPY requirements.txt /app/requirements.txt

RUN export TZ=Etc/UTC \
        && apt update --yes \
        && apt install g++ wget ffmpeg libsm6 libxext6 gimp libvulkan1 --yes \
        && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
        && dpkg -i cuda-keyring_1.1-1_all.deb \
        && rm -f cuda-keyring_1.1-1_all.deb \
        && apt update --yes \
        && apt install -y libcudnn8=8*-1+cuda11.8 libcudnn8-dev=8*-1+cuda11.8 \
        && pip install -r /app/requirements.txt \
        && apt remove g++ wget --yes \
        && apt autoremove --yes \
        && rm -rf /var/cache/apt

COPY . /app

# Prepare models
RUN python -u docker_prepare.py --continue-on-error

RUN rm -rf /tmp && mkdir /tmp && chmod 1777 /tmp

# Add /app to Python module path
ENV PYTHONPATH="/app"

WORKDIR /app

ENTRYPOINT ["python", "-m", "manga_translator"]