FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# Reuse the base image's CUDA packages without modifying its managed Python.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY requirements.txt /app/requirements.txt

# Update the base image's spin for huggingface_hub's newer Click requirement.
RUN export TZ=Etc/UTC \
        && apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
            g++ git python3-venv ffmpeg libsm6 libxext6 gimp libvulkan1 \
        && python -m venv --system-site-packages "${VIRTUAL_ENV}" \
        && "${VIRTUAL_ENV}/bin/python" -m pip install --no-cache-dir -r /app/requirements.txt spin==0.19 \
        && "${VIRTUAL_ENV}/bin/python" -m pip check \
        && apt-get remove g++ git --yes \
        && apt-get autoremove --yes \
        && rm -rf /var/lib/apt/lists/* /var/cache/apt

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
