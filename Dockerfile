FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    nano \
    vim \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Create user (safer than root)
RUN useradd -m -s /bin/bash developer && \
    chown -R developer:developer /opt/conda

USER developer
WORKDIR /home/developer

# Copy environment file
COPY --chown=developer:developer environments/environment_project.yml /home/developer/environment.yml

# Accept Conda Terms of Service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Download spaCy models
RUN /opt/conda/envs/talent_matching_linux/bin/python -m spacy download en_core_web_sm && \
    /opt/conda/envs/talent_matching_linux/bin/python -m spacy download en_core_web_lg

# Initialize conda and activate environment by default
RUN conda init bash && echo "conda activate talent_matching_linux" >> ~/.bashrc

WORKDIR /home/developer/project
CMD ["bash"]
