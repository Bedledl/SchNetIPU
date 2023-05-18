FROM docker.io/graphcore/pytorch:3.2.0

COPY requirements.txt /opt/requirements.txt

RUN mkdir ~/md_workdir

RUN apt-get update && apt-get install git -y

RUN pip3 install -r /opt/requirements.txt
