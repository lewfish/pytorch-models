FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN conda install -y notebook=6.1.* matplotlib=3.3.* scikit-learn=0.23.*
RUN apt-get update && apt-get install python-opengl && apt-get clean
RUN pip install gym==0.18.* pyvirtualdisplay==2.0 Box2D==2.3.* fastprogress==1.0.*

ENV PYTHONPATH=/opt/src/:$PYTHONPATH
COPY notebooks /opt/src/notebooks/
WORKDIR /opt/src

CMD ["bash"]
