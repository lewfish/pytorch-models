FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN conda install -y notebook=6.1.* matplotlib=3.3.* scikit-learn=0.23.*
RUN apt-get update && apt-get install -y python-opengl && apt-get clean
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
RUN pip install lightning-bolts==0.3.2

ENV PYTHONPATH=/opt/src/:$PYTHONPATH
COPY notebooks /opt/src/notebooks/
COPY pytorch_models /opt/src/pytorch_models/
WORKDIR /opt/src

CMD ["bash"]
