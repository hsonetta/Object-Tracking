from models import *
from utils import *
from torchvision import transforms

class detect():
    def __init__(self, config_path, weights_path, class_path, img_size, conf_thres, nms_thres, classes):
        self.config_path = config_path
        self.weights_path = weights_path
        self.class_path = class_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.classes = classes

        # Load model and weights
        self.model = Darknet(config_path, img_size=img_size)
        self.model.load_weights(weights_path)
        self.model.eval()
        self.Tensor = torch.FloatTensor

    def detect_image(self, img):
        ratio = min(self.img_size / img.size[0], self.img_size / img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                             transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0),
                                                             max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0)),
                                                            (128, 128, 128)),
                                             transforms.ToTensor(),
                                             ])
        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(self.Tensor))
        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = utils.non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)
        return detections[0]