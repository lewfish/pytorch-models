FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y wget=1.* python-opengl && apt-get clean

RUN wget -q -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /opt/conda/lib/:$LD_LIBRARY_PATH
RUN conda install -y python=3.7
RUN python -m pip install --upgrade pip

RUN conda install -y notebook=6.1.* matplotlib=3.3.* scikit-learn=0.23.*
RUN pip install torch==1.7.*
RUN pip install gym==0.18.* pyvirtualdisplay==2.0 Box2D==2.3.*

# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
RUN pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install torch-geometric

RUN pip install ipywidgets==7.6.3
RUN pip install pytorch-lightning==1.2.4
RUN pip install awscli==1.19.35 boto3==1.17.35
RUN pip install pytorch-lightning-bolts==0.3.2 opencv-python==4.5.1.48
RUN pip install torchvision==0.8.*
RUN pip install s3fs==0.6.0

# From https://vissl.readthedocs.io/en/v0.1.5/installation.html
RUN apt-get -y install git
RUN mkdir /tmp/vissl/ && \
    cd /tmp/vissl/ && \
    git clone --recursive https://github.com/facebookresearch/vissl.git && \
    cd vissl && \
    pip install --progress-bar off -r requirements.txt && \
    pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/master && \
    pip install -e .[dev]

ENV PYTHONPATH=/opt/src/:$PYTHONPATH
COPY notebooks /opt/src/notebooks/
COPY pytorch_models /opt/src/pytorch_models/
COPY configs /opt/src/configs/
WORKDIR /opt/src

CMD ["bash"]
