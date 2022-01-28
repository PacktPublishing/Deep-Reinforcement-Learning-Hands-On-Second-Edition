FROM python

RUN apt upgrade
RUN apt update
RUN pip install --upgrade pip

RUN apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python-opengl

ADD ./.devcontainer/pip-requirements.txt /tmp/
RUN pip install -r /tmp/pip-requirements.txt
RUN rm /tmp/pip-requirements.txt