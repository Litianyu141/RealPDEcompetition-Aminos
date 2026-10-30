FROM pytorch/pytorch@sha256:923f687790bec78081c357e71dcd5dcef80b0cc00f6c34484902a5e83362c854
RUN python -m pip install --no-cache-dir numpy==1.26.4 h5py==3.11.0 loguru==0.7.3 matplotlib==3.10.9
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
COPY . /workspace
