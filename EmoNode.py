#!/usr/bin/python3
import numpy as np
from pathlib import Path
import argparse

import torch
from emonet.models import EmoNet

import dlib

import rospy
import numpy as np

from visualization_msgs.msg import Marker
from pose_estimation.msg import Emotion
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge



image_size = 256
n_expression = 8
device = 'cpu'

class EmoNode:
    def __init__(self):
        torch.backends.cudnn.benchmark =  True

        # Parameters of the experiments
        self.expressions = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt', 8:'none'}

        # Loading the model 
        state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')

        print(f'Loading the model from {state_dict_path}.')
        state_dict = torch.load(str(state_dict_path), map_location='cpu')
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        self.net = EmoNet(n_expression=n_expression).to(device)
        self.net.load_state_dict(state_dict, strict=False)
        self.net.eval()
        print("Finished loading model")

        self.br = CvBridge()
        self.img = None
        self.img_sub = rospy.Subscriber("global_camera/compressed", CompressedImage, self.img_callback, queue_size=1, buff_size=52428800)

        self.emo_pub = rospy.Publisher("emotion/global", Emotion, queue_size = 10)
        self.viz_pub  = rospy.Publisher("/visualization_marker", Marker, queue_size=1)

        self.detector = dlib.get_frontal_face_detector()


    def img_callback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray, 1) # result
        if len(faces) == 0:
            return
        face = faces[0]
        x = face.left()
        y = face.top()
        x1 = face.right()
        y1 = face.bottom()

        w = x1 - x
        h = y1 - y
        cx = (x1 + x) / 2.0
        cy = (y1 + y) / 2.0

        size = max(w, h) * 1.1
        top = max(int(cy-size/2), 0)
        bottom = min(int(cy+size/2), img.shape[0])
        left = max(int(cx-size/2), 0)
        right = min(int(cx+size/2), img.shape[1])

        img = img[top:bottom, left:right, :]
        img = cv2.resize(img, (image_size,image_size), interpolation = cv2.INTER_AREA)
        
        # cv2.imshow("img", img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return

        img = img.transpose((2,0,1))

        img = np.expand_dims(img, 0)
        with torch.no_grad():
            x = torch.from_numpy(img)
            x = x.float()
            x /= 255.0
            out = self.net(x)

            expr = out['expression']
            exprs = np.squeeze(expr.cpu().numpy())
            expr = np.argmax(np.squeeze(expr.cpu().numpy()))

            val = out['valence']
            ar = out['arousal']

            val = np.squeeze(val.cpu().numpy())
            ar = np.squeeze(ar.cpu().numpy())

            emotion = Emotion()
            emotion.expressions = exprs
            emotion.expression = self.expressions[expr]
            emotion.valence = val
            emotion.arousal = ar

            self.emo_pub.publish(emotion)
            self.publish_text(self.expressions[expr])

            # print("Expression: ", self.expressions[expr])
            # print("Valence: ", val)
            # print("Arousal: ", ar)


    def publish_text(self, text):
        marker = Marker()

        marker.header.frame_id = "/human"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "emotion"

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3 ; Line strip 4
        marker.type = 9
        marker.id = 0

        marker.pose.position.x = 0.0
        marker.pose.position.y = -1.0
        marker.pose.position.z = 0.0    

        # Set the scale of the marker
        marker.scale.z = 0.25

        # Set the color
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = text

        self.viz_pub.publish(marker)
            
        
        
if __name__ == "__main__":

    rospy.init_node("emotion_node")

    emo_node = EmoNode()
    
    rospy.spin()