FROM nvidia/cuda:10.2-base-ubuntu16.04

LABEL maintainer "Dan Napierski (ISI) <dan.napierski@toptal.com>"

# Create app directory
WORKDIR /aida/src/

# Update
RUN apt-get update && apt-get install -y apt-utils wget bzip2 tree nano git unzip libgl1-mesa-glx
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh
RUN chmod +x ./Miniconda3-4.5.1-Linux-x86_64.sh
RUN ./Miniconda3-4.5.1-Linux-x86_64.sh -b -p ~/conda
ENV PATH="/root/conda/bin:${PATH}"
RUN echo $PATH
RUN conda update -n base -c defaults conda && conda -V && conda install setuptools && conda config --add channels conda-forge && conda install -c conda-forge rdflib lmdb

WORKDIR /aida/
RUN git clone https://github.com/tensorflow/models

WORKDIR /home/brian/facenet-master/
RUN git clone https://github.com/davidsandberg/facenet.git
RUN ls -al /home/brian/facenet-master/
ENV PYTHONPATH "/aida/src/:/aida/src/src/:/home/brian/facenet-master/:/aida/models/research/"
WORKDIR /aida/src/

COPY aida-env.txt ./
RUN conda create --name aida-env --file ./aida-env.txt python=3.6

WORKDIR	/aida/models/research/slim/
RUN /bin/bash -c ". activate aida-env && pip install -e ."

RUN tree -d -L 3 /aida/models/research/
RUN echo ${PYTHONPATH}

WORKDIR /aida/models/research/
RUN /bin/bash -c ". activate aida-env && protoc object_detection/protos/*.proto --python_out=."
#RUN /bin/bash -c ". activate aida-env && pip install -e ."

WORKDIR /aida/models/research/delf/
RUN /bin/bash -c ". activate aida-env && protoc delf/protos/*.proto --python_out=."
RUN /bin/bash -c ". activate aida-env && pip install -e ."

WORKDIR /aida/src/
RUN echo "source activate aida-env" >> ~/.bashrc
COPY check-imports.py ./
RUN /bin/bash -c ". activate aida-env && python check-imports.py"

# Bundle app source
COPY . .

WORKDIR /corpus/
ENV CORPUS="/corpus/"

# NOTE: ./.bigfiles/20180402-114759 should contain model files not in github
WORKDIR /models/facenet/
COPY ./.bigfiles/ .
RUN tree /models

WORKDIR /aida/src/columbia_recognition_models/
COPY ./.models/ .
# NOTE: ./.models/ needs to contain google500_2_classifier.pkl
RUN tree /aida/src/columbia_recognition_models
RUN tree /models

WORKDIR /output/
ENV OUTPUT="/output/"

WORKDIR /shared/cu_objdet_results/
ENV SHARED="/shared/"

WORKDIR /shared/cu_FFL_shared/jpg/jpg/
ENV JPG_PATH="/shared/cu_FFL_shared/jpg/jpg/"
ENV ZIP_PATH="/corpus/data/jpg/"
ENV LDCC_PATH="/corpus/data/jpg/jpg/"

WORKDIR /aida/src

LABEL name="AIDA Face and Building Detection"
LABEL version=0
LABEL revision=2

RUN chmod +x ./full_script.sh
CMD [ "/bin/bash", "" ]
