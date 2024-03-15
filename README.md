# Install:
- Azure Kinect SDK (read the repo exactly don't forget the USB install)
- Docker for Ubuntu
- Docker Nvidia
- Nvidia Driver + Cuda
- ROS 
- Python
- Git
- Git lfs
# Setup:
Copy the Azure Package into a ROS workspace and build them with `catkin_make`  
  
source data with `source ./devel/setup.bash` in your workspace folder  
  
make sure the camera is connected to a USB 3 Port and has power  
  
Build the the DockerFile in the DockerFiles folder `docker build -t yolo:1.0 .`
  
`mkdir container-data` in your home folder  
  
The launch files of the camera can be edited for example fps or resolution but also what topics should be published
# Start:
`roscore`  
  
start the azure package with `roslaunch azure_kinect_ros_driver driver.launch` or `roslaunch azure_kinect_ros_driver kinect_rgbd.launch`  
  
start docker with `docker run -it --net=host --gpus all -v /home/stefanl/container-data:/berry_pics yolo:1.0`
  
start bounding/sort algo with `rosrun yolorosort rosdetect.py`  

add another docker terminal, search for docker name `docker ps` and use `docker exec -it <container_name> bash` to open second terminal  
  
start cluster algo with `rosrun yolorosort cluster.py`

# Workflow Info
###### Docker
- show running container `docker ps`  
- show exited container `docker container ls --filter "status=exited” `
- delete container `docker rm <container_name>`
- show images `docker image list`
- delete image `docker image rm <imagename:version>`


