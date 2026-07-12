FROM apexnav:noetic

SHELL ["/bin/bash", "-lc"]

# CUDA-enabled PyTorch for RTX 3060. Host driver 580.x supports CUDA 12.x.
RUN /opt/micromamba/envs/apexnav/bin/python3 -m pip install --no-cache-dir --force-reinstall \
    torch==2.4.1+cu124 torchvision==0.19.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

RUN /opt/micromamba/envs/apexnav/bin/python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
    numpy==1.23.5 \
    transformers==4.30.2 \
    tokenizers==0.13.3 \
    huggingface-hub==0.16.4 \
    safetensors==0.3.1 \
    regex \
    addict \
    yapf \
    pycocotools \
    supervision==0.22.0 \
    defusedxml \
    platformdirs \
    pytz \
    python-dateutil \
    tzdata \
    pandas==2.2.2 \
    seaborn==0.13.2 \
    thop \
    timm \
    git+https://github.com/ChaoningZhang/MobileSAM.git
