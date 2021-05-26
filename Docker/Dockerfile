#FROM continuumio/miniconda:latest

FROM nvidia/cuda:11.3.0-cudnn8-runtime-centos8

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# 제작자 및 author 기입
LABEL maintainer="TEST"

# 해당 디렉토리에 있는 모든 하위항목들을 '/app`로 복사한다
COPY . /app

# image의 directory로 이동하고
WORKDIR /app

#miniconda 설치
RUN yum -y update && \
	yum install -y wget bzip2 curl git && \
	yum clean all

#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O
#RUN bash Miniconda3-latest-Linux-x86_64.sh
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh &&\
	/bin/bash ~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh && \
	/opt/conda/bin/conda clean -tipsy && \
	ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
	echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc

RUN /bin/bash -c "source ~/.bashrc"

# 필요한 의존성 file들 설치
RUN conda update -n base conda && \
	conda update --all && \
	conda create -n ai python=3.6 && \
	conda deactivate && \
	conda activate ai && \
	conda install pandas

# 환경 설정 세팅
#RUN python setup.py install

# container가 구동되면 실행
#ENTRYPOINT ["python", "Server.py"]

