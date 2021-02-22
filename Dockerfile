FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN conda install -y notebook=6.1.* matplotlib=3.3.* scikit-learn=0.23.*
RUN apt-get update && apt-get install -y python-opengl && apt-get clean
RUN pip install gym==0.18.* pyvirtualdisplay==2.0 Box2D==2.3.*

# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
RUN pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install torch-geometric

ENV PYTHONPATH=/opt/src/:$PYTHONPATH
COPY notebooks /opt/src/notebooks/
WORKDIR /opt/src

CMD ["bash"]
