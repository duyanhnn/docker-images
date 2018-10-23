FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:jonathonf/python-3.6 -y
RUN apt-get update -y
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN python3.6 -m pip install pip --upgrade
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    ca-certificates \
    gcc \
    git \
    libpq-dev \
    make \
    imagemagick \
    icu-devtools \
    libicu-dev \
    python3-tk \
    libsm6 \
    locales \
    libmecab-dev \
    mecab \
    mecab-ipadic-utf8 \
    libmysqlclient-dev \
    supervisor

RUN apt-get install curl -y
RUN apt-get install apt-transport-https -y
RUN curl http://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl http://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install msodbcsql17 -y
RUN apt-get install unixodbc-dev -y

RUN apt-get install libssl-dev libffi-dev -y
RUN mkdir /data /debug
WORKDIR /code/
# ADD . /code/

COPY new_req.txt /tmp
COPY supervisord.conf /etc/supervisor/supervisord.conf
RUN pip3 install numpy

RUN add-apt-repository ppa:alex-p/tesseract-ocr -y
RUN apt-get update -y
RUN apt-get install tesseract-ocr -y
# ADD ./Japanese.traineddata /usr/share/tesseract-ocr/4.00/tessdata/
# RUN export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

RUN pip3 install -r /tmp/new_req.txt

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get autoremove && apt-get clean
#CMD ["/bin/bash", "/code/entrypoint.sh"]
