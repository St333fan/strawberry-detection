# FROM Ubuntu
FROM ros:noetic-robot

# python install
RUN apt-get update
RUN yes | apt install python3-pip

#RUN apt-get update
#RUN apt-get install -y apt-utils

# openCV
RUN pip install opencv-python
RUN apt-get update ##[edited] braucht mann damit cv2 funktioniert im docker
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install tensorflow

# update
RUN apt-get update

# ros image bridge
RUN yes | sudo apt-get install ros-noetic-cv-bridge

# for cluster
RUN pip install -U scikit-learn			# for clustering

# for debugging
RUN apt-get install nano

COPY /resources  /

#RUN apt-get install python3-dateutil

# install requirements yolo and sort
WORKDIR /yolorosort/scripts 
RUN pip install -r requirements.txt
WORKDIR /yolorosort/scripts/sort
RUN pip install -r requirements.txt

# init ros workspace
WORKDIR /
RUN mkdir berry_ws
WORKDIR /berry_ws
RUN mkdir src
RUN . /opt/ros/noetic/setup.sh && catkin_make

# copy yolo package
WORKDIR /
RUN cp -r /yolorosort /berry_ws/src
WORKDIR /berry_ws
RUN . /opt/ros/noetic/setup.sh && catkin_make

# source bash file
RUN echo 'source /berry_ws/devel/setup.bash' >> ~/.bashrc
WORKDIR /
CMD ["/bin/bash"]
WORKDIR /berry_ws

# remendy yolo ERROR 
RUN cp -r /additionalFiles/. /usr/local/lib/python3.8/dist-packages/torch/nn/modules

