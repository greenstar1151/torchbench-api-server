# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3
# FROM nvcr.io/nvidia/l4t-base:r32.6.1

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV BENCHMARK_REPO https://github.com/greenstar1151/pytorch-benchmark
# ENV CONDA=https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-aarch64.sh
# ENV OPENBLAS_CORETYPE=ARMV8

# # Install basics
# RUN apt-get update
# RUN apt-get install git jq ffmpeg libsm6 libxext6 g++ -y

# # Install miniconda
# RUN wget "$CONDA" --no-check-certificate
# RUN chmod +x "$(basename $CONDA)"
# RUN ./"$(basename $CONDA)" -b -u

# ENV PATH="/root/miniconda3/bin:${PATH}"
# RUN activate; conda init; conda activate base; conda install -y python=3.6
# #RUN chmod +x ~/miniconda3/etc/profile.d/conda.sh; ~/miniconda3/etc/profile.d/conda.sh
# # RUN ["conda", "init", "bash"]

#RUN ~/miniconda3/etc/profile.d/conda.sh; conda activate base; conda install -y python=3.7

# apt
RUN apt-get update
# RUN apt-get install jq ffmpeg libsm6 libxext6 g++ liblapack-dev gfortran -y
RUN apt-get install git -y

# set up env for dependency install
RUN git clone -b main --single-branch ${BENCHMARK_REPO} /workspace/pytorch_benchmark
COPY ./pytorch_benchmark/install.py /workspace/pytorch_benchmark/install.py
COPY ./pytorch_benchmark/torchbenchmark/__init__.py /workspace/pytorch_benchmark/torchbenchmark/__init__.py
RUN python3 -m pip install --upgrade pip

# dependency list(additional per model):
#   alexnet -> None
#   mobilenet -> None
#   vgg16 -> None
#   BERT_pytorch -> BERT_pytorch/install.py
#   resnet -> None
#   attention_is_all_you_need_pytorch -> attention_is_all_you_need_pytorch/install.py

RUN pip3 install https://files.pythonhosted.org/packages/ff/87/c57d699a65acf6522cf28a589874746deb495ba73caff207bb7ec0399783/pandas-1.1.5-cp36-cp36m-manylinux2014_aarch64.whl
# #RUN pip3 install Cython numpy
# RUN pip3 install https://github.com/scipy/scipy/releases/download/v1.5.4/scipy-1.5.4-cp36-cp36m-manylinux2014_aarch64.whl

# RUN apt-get install python3-sklearn libaec-dev libblosc-dev libffi-dev libbrotli-dev libboost-all-dev libbz2-dev libgif-dev libopenjp2-7-dev liblcms2-dev libjpeg-dev libjxr-dev liblz4-dev liblzma-dev libpng-dev libsnappy-dev libwebp-dev libzopfli-dev libzstd-dev -y
# RUN pip3 install imagecodecs scikit-image

RUN cd /workspace/pytorch_benchmark; python3 install.py -v --continue_on_fail --install_whitelist alexnet mobilenet vgg16 BERT_pytorch resnet
#RUN conda install -y python=3.7

# # TEMP
# COPY ./pytorch_benchmark /workspace/pytorch_benchmark
# RUN cd /workspace/pytorch_benchmark; python3 install.py -v

# Install FastAPI dependency
COPY ./requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r /workspace/requirements.txt

# Run FastAPI server
WORKDIR /workspace
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload", "--reload-dir", "/workspace/app"]