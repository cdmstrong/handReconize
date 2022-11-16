#coding:utf-8
# date:2019-08
# Author: Eric.Lee
# function: predict camera
import argparse
import time
import os
import torch
from utils.datasets import *
from utils.utils import *
from utils.parse_config import parse_data_cfg
from utils.torch_utils import select_device
from yolov3.yolov3 import Yolov3, Yolov3Tiny
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def process_data(img, img_size=416):# 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img

def show_model_param(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        print("该层的结构: {}, 参数和: {}".format(str(list(i.size())), str(l)))
        k = k + l
    print("----------------------")
    print("总参数数量和: " + str(k))
def refine_hand_bbox(bbox,img_shape):
    height,width,_ = img_shape

    x1,y1,x2,y2 = bbox

    expand_w = (x2-x1)
    expand_h = (y2-y1)

    x1 -= expand_w*0.06
    y1 -= expand_h*0.1
    x2 += expand_w*0.06
    y2 += expand_h*0.1

    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

    x1 = int(max(0,x1))
    y1 = int(max(0,y1))
    x2 = int(min(x2,width-1))
    y2 = int(min(y2,height-1))

    return (x1,y1,x2,y2)

class DetectHand():
    def __init__(self) -> None:
        self.voc_config = './yolov3/cfg/hand.data' # 模型相关配置文件
        self.model_path = './yolov3/weights/hand-tiny_512-2021-02-19.pt' # 检测模型路径
        self.model_cfg = 'yolo-tiny' # yolo / yolo-tiny 模型结构
        self.video_path = "video/hand2.mp4" # 测试视频
        # 设备
        self.use_cuda = torch.cuda.is_available()
        self.device = select_device()
        self.img_size = 416 # 图像尺寸
        self.conf_thres = 0.5# 检测置信度
        self.nms_thres = 0.6 # nms 阈值
        self.classes = load_classes(parse_data_cfg(self.voc_config)['names'])
        self.num_classes = len(self.classes)
        self.model_init()
        
    def model_init(self):
        
        if "-tiny" in self.model_cfg:
            a_scalse = 416./self.img_size
            anchors=[(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]
            anchors_new = [ (int(anchors[j][0]/a_scalse),int(anchors[j][1]/a_scalse)) for j in range(len(anchors)) ]

            self.model = Yolov3Tiny(self.num_classes,anchors = anchors_new)

        else:
            a_scalse = 416./self.img_size
            anchors=[(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)]
            anchors_new = [ (int(anchors[j][0]/a_scalse),int(anchors[j][1]/a_scalse)) for j in range(len(anchors)) ]
            self.model = Yolov3(self.num_classes,anchors = anchors_new)
        # Initialize model
        # Load weights
        weights = self.model_path
        if os.access(weights,os.F_OK):# 判断模型文件是否存在
            self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        else:
            print('error model not exists')
            return False
        self.model.to(self.device).eval()#模型模式设置为 eval
        show_model_param(self.model)# 显示模型参数
        
    def detect(self, im0):
            t = time.time()
            # im0 = cv2.imread("samples/5.png")
            img = process_data(im0, self.img_size)
            if self.use_cuda:
                torch.cuda.synchronize()
            t1 = time.time()
            print("process time:", t1-t)
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)
            self.model.to(self.device).eval()#模型模式设置为 eval
            with torch.no_grad():
                pred, _ = self.model(img)#图片检测
            if self.use_cuda:
                torch.cuda.synchronize()
            t2 = time.time()
            # print("inference time:", t2-t1)
            detections = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0] # nms
            if self.use_cuda:
                torch.cuda.synchronize()
            t3 = time.time()
            # print("get res time:", t3-t2)
            if detections is None or len(detections) == 0:
                return None
            # Rescale boxes from 416 to true image size
            detections[:, :4] = scale_coords(self.img_size, detections[:, :4], im0.shape).round()
            result = []
            for res in detections:
                result.append((self.classes[int(res[-1])], float(res[4]), [int(res[0]), int(res[1]), int(res[2]), int(res[3])]))
            if self.use_cuda:
                torch.cuda.synchronize()

            # print(result)

            for r in result:
                print(r)

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in detections:
                label = '%s %.2f' % (self.classes[int(cls)], conf)
                # label = '%s' % (classes[int(cls)])

                # print(conf, cls_conf)
                # xyxy = refine_hand_bbox(xyxy,im0.shape)
                xyxy = int(xyxy[0]),int(xyxy[1])+6,int(xyxy[2]),int(xyxy[3])
                if int(cls) == 0:
                    plot_one_box(xyxy, im0, label=label, color=(15,255,95),line_thickness = 3)
                else:
                    plot_one_box(xyxy, im0, label=label, color=(15,155,255),line_thickness = 3)

            s2 = time.time()
            # print("detect time: {} \n".format(s2 - t))

            str_fps = ("{:.2f} Fps".format(1./(s2 - t+0.00001)))
            cv2.putText(im0, str_fps, (5,im0.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 255),4)
            cv2.putText(im0, str_fps, (5,im0.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0),1)
            return im0
if __name__ == '__main__':

    voc_config = './yolov3/cfg/hand.data' # 模型相关配置文件
    model_path = './yolov3/weights/hand_416-2021-02-20.pt' # 检测模型路径
    model_cfg = 'yolo' # yolo / yolo-tiny 模型结构
    video_path = "./video/hand2.mp4" # 测试视频

    img_size = 416 # 图像尺寸
    conf_thres = 0.5# 检测置信度
    nms_thres = 0.6 # nms 阈值
