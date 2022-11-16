import os
import sys

from yolov3.predict import DetectHand
sys.path.append(r"./video") 
import argparse
from models.rexnetv1 import ReXNetV1
import torch
import cv2
import numpy as np

class Video_inference():
    def __init__(self, opt) -> None:
        self.video_path = opt.video_path
        self.save = opt.save
        self.model = ReXNetV1( width_mult=1.0, depth_mult=1.0, num_classes=opt.num_classes)
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
         # 读取模型参数
        dict = torch.load(opt.model_dict, map_location=torch.device(device))
        self.model.load_state_dict(dict)
        self.model.to(device)
        self.DetectHand = DetectHand()
        self.startAngle = 0
        self.endAngle = 30
        self.statePos = (0, 0)
    def detect_img(self, img: cv2.Mat):
        self.model.eval()
        with torch.no_grad():
            print(img.shape)
            img_width = img.shape[1]
            img_height = img.shape[0]
            img_ = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
            img_ = img_.astype(np.float32)
            
            img_ = (img_ - 128)/256
            img_ = img_.transpose(2, 0, 1)
            
            img_: torch.Tensor = torch.from_numpy(img_)
            img_ = img_.unsqueeze(0)
            
            pre = self.model(img_.float())
            output = np.squeeze(pre.numpy())
            
            pts_hand = {}
            
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))

                pts_hand[str(i)] = {}
                pts_hand[str(i)] = {
                    "x":x,
                    "y":y,
                    }
            draw_bd_handpose(img,pts_hand,0,0) # 绘制关键点连线
            
            #------------- 绘制关键点
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))

                cv2.circle(img, (int(x),int(y)), 3, (255,50,60),-1)
                cv2.circle(img, (int(x),int(y)), 1, (255,150,180),-1)
                if i == 4:
                    if np.abs((x - self.statePos[0])) <= 30 and np.abs((y - self.statePos[1])) <= 30:
                        cv2.ellipse(img, (int(x),int(y)), (50, 50), 0, self.startAngle, self.endAngle, (0, 255, 128), 3)
                        # self.startAngle = self.endAngle
                        self.endAngle += 30
                    else:
                        self.endAngle = 30
                    self.statePos = (int(x), int(y))
            return img

    def detect_video(self):
        video = cv2.VideoCapture(self.video_path)
        while(video.isOpened()):
            ret, frame = video.read()
            if ret == False:
                break
            frame = resize(frame, height=frame.shape[0]/3)
            print(frame.shape)
            hand =  self.DetectHand.detect(frame)
            if hand is not None:
                img = self.detect_img(hand)
            else:
                img = frame
            cv2.imshow('pre', img)
            key = cv2.waitKey(60)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
        
def resize(img, width = None, height = None):
    w = img.shape[1]
    h = img.shape[0]
    rateH = 1
    rateW = 1
    if width:
        rateH = width/w
        height = h * rateH
    elif height:
        rateW = height/h
        width = w * rateW
    img = cv2.resize(img, (int(width), int(height)))
    return img
    
            
    
def draw_bd_handpose(img_,hand_,x,y):
    thick = 2
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55)]
    #
    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['1']['x']+x), int(hand_['1']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['1']['x']+x), int(hand_['1']['y']+y)),(int(hand_['2']['x']+x), int(hand_['2']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['2']['x']+x), int(hand_['2']['y']+y)),(int(hand_['3']['x']+x), int(hand_['3']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['3']['x']+x), int(hand_['3']['y']+y)),(int(hand_['4']['x']+x), int(hand_['4']['y']+y)), colors[0], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['5']['x']+x), int(hand_['5']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['5']['x']+x), int(hand_['5']['y']+y)),(int(hand_['6']['x']+x), int(hand_['6']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['6']['x']+x), int(hand_['6']['y']+y)),(int(hand_['7']['x']+x), int(hand_['7']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['7']['x']+x), int(hand_['7']['y']+y)),(int(hand_['8']['x']+x), int(hand_['8']['y']+y)), colors[1], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['9']['x']+x), int(hand_['9']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['9']['x']+x), int(hand_['9']['y']+y)),(int(hand_['10']['x']+x), int(hand_['10']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['10']['x']+x), int(hand_['10']['y']+y)),(int(hand_['11']['x']+x), int(hand_['11']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['11']['x']+x), int(hand_['11']['y']+y)),(int(hand_['12']['x']+x), int(hand_['12']['y']+y)), colors[2], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['13']['x']+x), int(hand_['13']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['13']['x']+x), int(hand_['13']['y']+y)),(int(hand_['14']['x']+x), int(hand_['14']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['14']['x']+x), int(hand_['14']['y']+y)),(int(hand_['15']['x']+x), int(hand_['15']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['15']['x']+x), int(hand_['15']['y']+y)),(int(hand_['16']['x']+x), int(hand_['16']['y']+y)), colors[3], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['17']['x']+x), int(hand_['17']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['17']['x']+x), int(hand_['17']['y']+y)),(int(hand_['18']['x']+x), int(hand_['18']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['18']['x']+x), int(hand_['18']['y']+y)),(int(hand_['19']['x']+x), int(hand_['19']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['19']['x']+x), int(hand_['19']['y']+y)),(int(hand_['20']['x']+x), int(hand_['20']['y']+y)), colors[4], thick)
   
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=' Project Hand Pose')
    parser.add_argument("--video_path", type=str, default="./video/myHand2.mp4") 
    parser.add_argument('--save', default=False)
    parser.add_argument('--model', default="ReXNetV1")
    parser.add_argument('--model_dict', default="./weights/ReXNetV1-size-256-loss-wing_loss102-0.115-20211108.pth")
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size')
    parser.add_argument('--num_classes', type=int , default = 42,
        help = 'num_classes') #  手部21关键点， (x,y)*2 = 42
    opt = parser.parse_args()
    
    print('\n/******************* {} ******************/\n'.format(parser.description))
    
    unparse = vars(opt)
    for key in unparse.keys():
        print('{} : {}'.format(key,unparse[key]))
        
    print('.........................>>>>>>>>>>>>>>>>>............')
    
    inference = Video_inference(opt)
    inference.detect_video()
    # img = inference.detect_img(cv2.imread('./video/hand.png'))
    # cv2.imshow('h', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    
    