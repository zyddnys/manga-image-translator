FROM pytorch/pytorch:latest

WORKDIR /app

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
# Assume root to install required dependencies
RUN apt-get install -y git g++ ffmpeg libsm6 libxext6 libvulkan-dev

# Install pip dependencies

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt
RUN pip install torchvision --force-reinstall
RUN pip install "numpy<2.0"

RUN apt-get remove -y g++ && \
    apt-get autoremove -y

# Copy app
COPY . /app

# Prepare models
RUN python -u docker_prepare.py

RUN rm -rf /tmp

# Add /app to Python module path
ENV PYTHONPATH="${PYTHONPATH}:/app"

WORKDIR /app

ENTRYPOINT ["python", "-m", "manga_translator"]
