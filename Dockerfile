FROM pytorch/pytorch:latest

WORKDIR /app

# Assume root to install required dependencies
RUN apt-get update && \
    apt-get install -y git g++ ffmpeg libsm6 libxext6 libvulkan-dev && \
    pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# Install pip dependencies

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt
RUN pip install torchvision --force-reinstall

RUN apt-get remove -y g++ && \
    apt-get autoremove -y

# Copy app
COPY . /app

# Prepare models
RUN python -u docker_prepare.py

ENTRYPOINT ["python", "-u", "/app/translate_demo.py"]
