FROM nvidia/cuda:10.2-base-ubuntu16.04

LABEL maintainer "Dan Napierski (ISI) <dan.napierski@toptal.com>"

# Create app directory
WORKDIR /aida/src/

# Update
RUN apt-get update && apt-get install -y apt-utils wget bzip2 tree nano git 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh
RUN chmod +x ./Miniconda3-4.5.1-Linux-x86_64.sh
RUN ./Miniconda3-4.5.1-Linux-x86_64.sh -b -p ~/conda
ENV PATH="/root/conda/bin:${PATH}"
RUN echo $PATH
RUN conda update -n base -c defaults conda && conda -V && conda install setuptools

WORKDIR /home/brian/facenet-master/
RUN git clone https://github.com/davidsandberg/facenet.git
RUN ls -al /home/brian/facenet-master/

WORKDIR /object-detection/src/lib
RUN git clone --branch v1.12.0 https://github.com/tensorflow/models.git
RUN git clone --branch tag/v1.0.3 https://github.com/NextCenturyCorporation/AIDA-Interchange-Format.git

ENV PYTHONPATH "/aida/src/:/aida/src/slim/:/aida/src/src/:/home/brian/facenet-master/:/usr/local/bin/python:/object-detection/src/lib/models/research:/object-detection/src/lib/models/research/slim:/object-detection/src/lib/AIDA-Interchange-Format/python:."
WORKDIR /aida/src/

COPY aida-env.txt ./
RUN conda create --name aida-env --file ./aida-env.txt -c conda-forge tensorflow-gpu=1.14 rdflib=4.2.2 python=3.6
RUN echo "source activate aida-env" >> ~/.bashrc

# Bundle app source
COPY . .

WORKDIR /corpus/
ENV CORPUS="/corpus/"

# NOTE: ./.bigfiles/20180402-114759 should contain model files not in github
WORKDIR /models/facenet/
COPY ./.bigfiles/ .
ENV MODELS="/models/"

WORKDIR /output/
ENV OUTPUT="/output/"

WORKDIR /shared/cu_objdet_results/
ENV SHARED="/shared/"

WORKDIR /shared/cu_FFL_shared/jpg/jpg/
ENV JPG_PATH="/shared/cu_FFL_shared/jpg/jpg/"
ENV ZIP_PATH="/corpus/data/jpg/"
# TODO extract HC0000.jpg.zip out of ZIP_PATH
ENV LDCC_PATH="/corpus/data/jpg/jpg/"

WORKDIR /aida/src

RUN apt-get -y install unzip
#'*.jpg.ldcc'):

LABEL name="AIDA Face and Building Detection"
LABEL version=0
LABEL revision=1

# Open port
EXPOSE 8082
CMD [ "/bin/bash", "" ]
