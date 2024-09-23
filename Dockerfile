# Use the conda-forge base image with Python
FROM mambaorg/micromamba:1.5.8


# set environment variables
ENV PYTHONUNBUFFERED 1

RUN micromamba config append channels conda-forge
RUN micromamba config append channels openeye

COPY --chown=$MAMBA_USER:$MAMBA_USER  environment.yaml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER  .  /home/mambauser/FALCBot

RUN micromamba install -y -n base git -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /home/mambauser/asap-ml-streamlit

RUN mkdir /home/mambauser/.OpenEye
ENV OE_LICENSE=/home/mambauser/.OpenEye/oe_license.txt