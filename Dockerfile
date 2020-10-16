FROM ubuntu:latest
MAINTAINER Abhijit Roy

RUN apt-get update -y

RUN apt-get install -y python3-pip python3-dev build-essential

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD python3 model_deployer.py
