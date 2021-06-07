from models import *
from utils import *
from PyTorch_Object_Detection import detect
from PyTorch_Object_Tracking import track

videopath = r'Object Tracking/data/video/dash_cam_2.mp4'
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
classes = utils.load_classes(class_path)

object_det = detect(config_path, weights_path, class_path, img_size, conf_thres, nms_thres, classes)

object_track = track(object_det, videopath, img_size, classes)

object_track.track_object()


