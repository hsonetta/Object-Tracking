from models import *
from utils import *
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sort import *

class track():
    def __init__(self, object_detect, videopath, img_size, classes):
        self.object_detect = object_detect
        self.videopath = videopath
        self.img_size = img_size
        self.classes = classes

    def track_object(self):
        cmap = plt.get_cmap('tab20b')
        colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

        # initialize Sort object and video capture
        vid = cv2.VideoCapture(self.videopath)
        mot_tracker = Sort()

        cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Stream', (800, 600))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        ret, frame = vid.read()
        vw = frame.shape[1]
        vh = frame.shape[0]
        print("Video size", vw, vh)
        outvideo = cv2.VideoWriter(self.videopath.replace(".mp4", "-det.mp4"), fourcc, 20.0, (vw, vh))

        frames = 0
        starttime = time.time()
        while (True):
            ret, frame = vid.read()
            if not ret:
                break
            frames += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(frame)
            detections = self.object_detect.detect_image(pilimg)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = np.array(pilimg)
            pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
            unpad_h = self.img_size - pad_y
            unpad_w = self.img_size - pad_x
            if detections is not None:
                tracked_objects = mot_tracker.update(detections.cpu())

                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                    color = colors[int(obj_id) % len(colors)]
                    cls = self.classes[int(cls_pred)]
                    cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
                    cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 80, y1), color, -1)
                    cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 3)

            cv2.imshow('Stream', frame)
            outvideo.write(frame)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        totaltime = time.time() - starttime
        print(frames, "frames", totaltime / frames, "s/frame")
        cv2.destroyAllWindows()
        outvideo.release()