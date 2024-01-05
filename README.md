# YOLOX-ByteTrack-Any-Class-Counter

## Getting started
### Docker
```shell
docker build -t bytetrack:latest .

# Startup sample
mkdir -p pretrained && \
mkdir -p YOLOX_outputs && \
xhost +local: && \
docker run --gpus all -it --rm \
-v $PWD/pretrained:/workspace/ByteTrack/pretrained \
-v $PWD/datasets:/workspace/ByteTrack/datasets \
-v $PWD/YOLOX_outputs:/workspace/ByteTrack/YOLOX_outputs \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
bytetrack:latest
```
## running demo 
```shell
git clone https://github.com/cudanexus/yolox-bytetrack.git
cd yolox-bytetrack
# Change the files as required per classes filters etc
python detector.py  

```

### Pip
```bash
pip install -r requirements.txt
cd YOLOX
pip install -v -e .
```
### Nvidia Driver
Make sure to use CUDA Toolkit version 11.2 as it is the proper version for the Torch version used in this repository: https://developer.nvidia.com/cuda-11.2.0-download-archive

### torch2trt
Clone this repository and install: https://github.com/NVIDIA-AI-IOT/torch2trt 

## Download a pretrained model
Download pretrained yolox_s.pth file: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth

Copy and paste yolox_s.pth from your downloads folder into the 'YOLOX' folder of this repository.

## Convert model to TensorRT
```bash
python tools/trt.py -n yolox-s -c yolox_s.pth
```

## Runing the Counter with YOLOX-s
In file detector.py you need to replace the file video name in line 131:. 

Then run this command:
```bash
python detector.py
```

## References
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)

