FROM pytorch/pytorch:latest

ARG RELEASE_VERSION=beta-0.3
ARG ASSET_BASE_URL=https://github.com/zyddnys/manga-image-translator/releases/download

WORKDIR /app

# Assume root to install required dependencies
RUN apt-get update && \
    apt-get install -y git g++ ffmpeg libsm6 libxext6 libvulkan-dev && \
    pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# Install pip dependencies

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt
RUN pip install torchvision --force-reinstall

RUN apt-get remove -y g++ && apt-get autoremove -y

# Copy app
COPY . /app

# Prepare models
RUN python -u docker_prepare.py

# Copy remaing dependencies
ADD ${ASSET_BASE_URL}/${RELEASE_VERSION}/detect.ckpt /app/
ADD ${ASSET_BASE_URL}/${RELEASE_VERSION}/comictextdetector.pt /app/
ADD ${ASSET_BASE_URL}/${RELEASE_VERSION}/comictextdetector.pt.onnx /app/

# Remove cache
RUN rm -rf /tmp/*

ENTRYPOINT ["python", "-u", "/app/translate_demo.py"]
