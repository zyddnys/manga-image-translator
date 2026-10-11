FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install pip dependencies

COPY requirements.txt /app/requirements.txt

RUN export TZ=Etc/UTC \
        && apt update --yes \
        && apt install g++ git ffmpeg libsm6 libxext6 gimp libvulkan1 --yes \
        && pip install -r /app/requirements.txt \
        && apt remove g++ git --yes \
        && apt autoremove --yes \
        && rm -rf /var/cache/apt

COPY . /app

# Prepare models
RUN --mount=type=cache,target=/.cache/models \
    cp -rn /.cache/models/. /app/models/ || true && \
    python -u docker_prepare.py --continue-on-error && \
    cp -rn /app/models/. /.cache/models/ || true

RUN rm -rf /tmp && mkdir /tmp && chmod 1777 /tmp

# Add /app to Python module path
ENV PYTHONPATH="/app"

WORKDIR /app

ENTRYPOINT ["python", "-m", "manga_translator"]
