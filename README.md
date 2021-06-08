# Object Detection and Tracking using PyTorch

Pretrained Yolo model is used to detect objects in the video frames. The objects across the frames are tracked using SORT. Watch demo in the following video.

[![Watch the video](https://i.imgur.com/MCWaE9Z.png)](https://youtu.be/FmzJZ7hJfHA)

Object tracking identifies a specific object over time and tells us if an object in one frame is the same as one in a previous frame. Object tracing is used in various applications like Aerial object tracking, Industrial automation (Counting objects on a conveyor belt), Military defense, etc. This repository enables taking a video file as an input using OpenCV and will generate a resulting annotation file. 

References:

1. YOLOv3: https://pjreddie.com/darknet/yolo/
2. SORT paper: https://arxiv.org/pdf/1602.00763.pdf
