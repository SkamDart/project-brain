
from fabric.api import *


def tf_docker_aws():
	"""
	AWS_GPU Env setup for TF

	Usage: 
		fab tf_docker_aws -i ~/path/to/aws.pem user@hostname
	"""	
	# setup nvidia drivers
	sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/
	x86_64/7fa2af80.pub

	sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64 /
	" > /etc/apt/sources.list.d/cuda.list'

	sudo apt-get update && sudo apt-get install -y --no-install-recommends cuda-drivers

	# install latest docker
	sudo apt-get update
	sudo curl -fsSL https://get.docker.com/ | sh
	sudo curl -fsSL https://get.docker.com/gpg | sudo apt-key add -

	# setup nvidia docker
	wget https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-rc.3/nvidia-docker_1.0.0.rc
	.3-1_amd64.deb
	sudo dpkg -i nvidia-docker_1.0.0.rc.3-1_amd64.deb
	sudo rm nvidia-docker_1.0.0.rc.3-1_amd64.deb

	# download docker image
	sudo docker pull fluxcapacitor/gpu-tensorflow

	# start tf-gpu docker container
	sudo nvidia-docker run -itd --name=gpu-tensorflow -e "PASSWORD=password" -p 8754:8888 -p 6006:600
	6 fluxcapacitor/gpu-tensorflow

	"""
	verify successful startup:
		> sudo nvidia-docker exec -it gpu-tensorflow bash	
		> nvidia-smi
	
	Look for GPU fan column in output
	"""
