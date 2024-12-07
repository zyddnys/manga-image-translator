FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# not apt update: most effective code in pytorch base image is in /opt/conda

WORKDIR /app

# Install pip dependencies

COPY requirements.txt /app/requirements.txt

RUN export TZ=Etc/UTC ; \
        apt update --yes \
        && apt install g++ ffmpeg libsm6 libxext6 --yes \
        && pip install -r /app/requirements.txt \
        && apt remove g++ --yes \
        && apt autoremove --yes \
        && rm -rf /var/cache/apt

COPY . /app

# Prepare models
RUN python -u docker_prepare.py --continue-on-error

RUN rm -rf /tmp

# Add /app to Python module path
ENV PYTHONPATH="/app"

WORKDIR /app

ENTRYPOINT ["python", "-m", "manga_translator"]
