#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import serial
import time


# In[2]:


# 변수 지정
fire_prob = 0.2  #fire일 확률 0 ~ 1

left_slope_limit_low = -80   #left line limit, negative
left_slope_limit_high = -1  #left line limit, negative
right_slope_limit_low = 1   #right line limit, positive
right_slope_limit_high = 80  #right line limit, positive

threshold_top_fixel = 100   #threshold fixel for homography

canny_low=100#100
canny_high=250#250
houghlimit=120#90
#corridor area
area_x=590  
area_y=1000


# In[3]:


# homography calculation function
def afterx(a,b,h):
    return (h[0][0]*a + h[0][1]*b + h[0][2]) / ((h[2][0]*a + h[2][1]*b + h[2][2]))

def aftery(a,b,h):
    return (h[1][0]*a + h[1][1]*b + h[1][2]) / ((h[2][0]*a + h[2][1]*b + h[2][2]))

# 입력 사이즈 리스트 (Yolo 에서 사용되는 네크워크 입력 이미지 사이즈)
size_list = [320, 416, 608]


# In[4]:


def yolo(frame, size, score_threshold, nms_threshold):
    # YOLO 네트워크 불러오기
    net = cv2.dnn.readNet("yolov3_5500.weights", "yolov3.cfg")
    classes = []

    with open("obj.names","rt",encoding="UTF8") as f:
        classes=[line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # 클래스의 갯수만큼 랜덤 RGB 배열을 생성
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 이미지의 높이, 너비, 채널 받아오기
    height, width, channels = frame.shape

    # 네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (size, size), (0, 0, 0), True, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(blob)

    # 결과 받아오기
    outs = net.forward(output_layers)

    # 각각의 데이터를 저장할 빈 리스트
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > fire_prob:
          # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

          # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    print("================================start===============================")
    # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
    print(f"boxes: {boxes}")
    print(f"confidences: {confidences}")

    # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)
    
    # 후보 박스 중 선택된 박스의 인덱스 출력
    print(f"indexes: ", end='')
    for index in indexes:
        print(index, end=' ')
    print("\n\n============================== classes ==============================")

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]]
            label = f"{class_name} {confidences[i]:.2f}"
            color = colors[class_ids[i]]

        # 사각형 테두리 그리기 및 텍스트 쓰기
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
           
        # 탐지된 객체의 정보 출력
            print(f"[{class_name}({i})] conf: {confidences[i]} / x: {x} / y: {y} / width: {w} / height: {h}")
            
    return frame, center_x, center_y, height, width


# In[5]:


def line_detection(frame, center_x, center_y, height, width):

    center = width / 2

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0, 0)
    edges = cv2.Canny(blur,canny_low, canny_high,apertureSize = 3) 

    lines = cv2.HoughLines(edges,0.8,np.pi/180, houghlimit)
    
    cv2.circle(frame,(center_x,center_y), 10, (0,0,255),-1)


    for i in lines:
        r,theta = i[0][0], i[0][1]
        a = np.cos(theta)
        b = np.sin(theta)

      # x0 stores the value rcos(theta)
        x0 = a*r

      # y0 stores the value rsin(theta)
        y0 = b*r

      # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))
      # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))
      # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))
      # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))
     

     
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.

      # slope calculate
        if ((x2-x1) == 0):
            slope = 0
        else:
            slope = (y2-y1)/(x2-x1)

      # intercept calculate
        intercept = y1 - slope * x1
            
  # Draw line, two point for homography

      # left
        if (slope > left_slope_limit_low) and (slope < left_slope_limit_high):
            cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 1)

            left_slope = slope
            left_intercept = intercept

            left_x_bottom = (height - left_intercept) / left_slope
            left_x_top = (threshold_top_fixel - left_intercept) / left_slope

            load_left_test_y = left_slope * center_x + left_intercept
            
      # right
        if (slope > right_slope_limit_low) and (slope < right_slope_limit_high):
            cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 1)
        
            right_slope = slope
            right_intercept = intercept

            right_x_bottom = (height - right_intercept) / right_slope
            right_x_top = (threshold_top_fixel - right_intercept) / right_slope

            load_right_test_y = right_slope * center_x + right_intercept
            

      # fire position 
        if (center_x < center) :
            if center_y >= load_left_test_y: 
                projection_y = center_y
            else:
                projection_y = (left_slope * center_x) + left_intercept

        elif (center_x >= center) :
            if center_y >= load_right_test_y: 
                projection_y = center_y
            else:
                projection_y = (right_slope * center_x) + right_intercept         

        
    cv2.circle(frame, (int(center_x), int(projection_y)), 10, (0,0,255), -1)    

    return frame, center_x, projection_y, left_x_top, right_x_top, left_x_bottom, right_x_bottom


# In[6]:


def f_homography(frame, homo_img, left_x_top, right_x_top, left_x_bottom, right_x_bottom, height, center_x, projection_y):
    pts_src = np.array([[left_x_top,threshold_top_fixel], [right_x_top, threshold_top_fixel], [left_x_bottom, height], [right_x_bottom, height]])
    pts_dst = np.array([[0,0],[589,0],[0,999],[589,999]])

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(frame, h, (homo_img.shape[1], homo_img.shape[0]))

    pos_x = afterx(center_x, projection_y, h)
    pos_y = aftery(center_x, projection_y, h)

    cv2.circle(im_out, (int(pos_x), int(pos_y)), 10, (0, 255, 0), -1)

    return im_out, pos_x, pos_y


# In[7]:


#calculate angle & Python to Arduino

def send_arduino(pos_x, pos_y):

    sprinkler_2_x=area_x*0.5
    sprinkler_2_y=area_y*0.75

    angle_2=math.atan2(pos_x-sprinkler_2_x, sprinkler_2_y-pos_y)*180/math.pi

    length_2=math.sqrt(pow(pos_x-sprinkler_2_x,2)+pow(sprinkler_2_y-pos_y,2))

    if angle_2<0:
        angle_2+=360
        
    if (angle_2<10):
        a="Q" + str(int(length_2)) + "00" + str(int(angle_2))

    elif (angle_2<100):
        a="Q"+str(int(length_2)) + "0" +str(int(angle_2))
    
    else:
        a="Q"+str(int(length_2)) + str(int(angle_2))
    
    print("------------------")
    print("angle & length")
    print(angle_2)
    print(length_2)

    print(a)
    
   # c= a.encode('utf-8')
   # arduino.write(c)
    


# In[8]:


cap = cv2.VideoCapture("fire_right_top.mp4")
homo_img = cv2.imread("homo_img.png")
state_f=0
state_l=0
#arduino = serial.Serial('COM5', 9600)

if cap.isOpened():
    while True:
        ret, img=cap.read()
        
        img = cv2.flip(img,0)
        #img= cv2.flip(img,1)
        
        try:
            frame, center_x, center_y, height, width = yolo(frame=img, size=size_list[1], score_threshold=0.4, nms_threshold=0.4)
            print("--------------------------")
            print("center")
            print(center_x, center_y)
            state_f=0
        except:
            print("fire detection error")
            frame = img
            state_f=1
            
        try:
            frame, center_x, projection_y, left_x_top, right_x_top, left_x_bottom, right_x_bottom = line_detection(frame, center_x, center_y, height, width)
            print("-----------------")
            print("projection")
            print(center_x, projection_y)
            state_l=0
        except:
            print("line detection error")
            frame = img
            state_l=1
            
        try:
            frame, pos_x, pos_y = f_homography(frame, homo_img, left_x_top, right_x_top, left_x_bottom, right_x_bottom, height, center_x, projection_y)
            
            print("--------------------------")
            print("pos")
            print(pos_x, pos_y)
        except:
            print("Homography error")
            frame = img
        
#         try:
#             #if (state_f==0) and (state_l==0):
#             send_arduino(pos_x, pos_y)
#         except:
#             print("arduino send error")
            
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    
        if ret:
            
            cv2.imshow("result", frame)
            
            cv2.waitKey(0)

else:
    print("can't open video.")

cap.release()
cv2.destroyAllWindows()




#test code

import cv2
import numpy as np
from matplotlib import pyplot as plt
#from google.colab.patches import cv2_imshow
import math
import serial


# In[14]:


# 변수 지정
fire_prob = 0.2  #fire일 확률 0 ~ 1

left_slope_limit_low = -89   #left line limit, negative
left_slope_limit_high = -1  #left line limit, negative
right_slope_limit_low = 1   #right line limit, positive
right_slope_limit_high = 89  #right line limit, positive

threshold_top_fixel = 400   #threshold fixel for homography

#corridor area
area_x=590  
area_y=1000


# In[15]:


def line_detection(frame, center_x, center_y, height, width):

    center =  width / 2
    state_L=0
    state_H=0

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0, 0)
    edges = cv2.Canny(blur,30,250,apertureSize = 3) 

    lines = cv2.HoughLines(edges,0.8,np.pi/180, 100)
    cv2.circle(frame,(center_x,center_y), 10, (0,0,255),-1)


    for i in lines:
        r,theta = i[0][0], i[0][1]
        a = np.cos(theta)
        b = np.sin(theta)

      # x0 stores the value rcos(theta)
        x0 = a*r

      # y0 stores the value rsin(theta)
        y0 = b*r

      # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))
      # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))
      # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))
      # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))
     

     
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.

      # slope calculate
        if ((x2-x1) == 0):
            slope = 0
        else:
            slope = (y2-y1)/(x2-x1)
        print(slope)
      # intercept calculate
        intercept = y1 - slope * x1
            
  # Draw line, two point for homography

      # left
        if (slope > left_slope_limit_low) and (slope < left_slope_limit_high):
            cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 1)

            left_slope = slope
            left_intercept = intercept

            left_x_bottom = (height - left_intercept) / left_slope
            left_x_top = (threshold_top_fixel - left_intercept) / left_slope

            load_left_test_y = left_slope * center_x + left_intercept
            state_L=1

      # right
        if (slope > right_slope_limit_low) and (slope < right_slope_limit_high):
            cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 1)
        
            right_slope = slope
            right_intercept = intercept

            right_x_bottom = (height - right_intercept) / right_slope
            right_x_top = (threshold_top_fixel - right_intercept) / right_slope

            load_right_test_y = right_slope * center_x + right_intercept
            state_R=1
      # Draw Point : fire, line, projection position
    #cv2.circle(frame, (int(center_x), int(center_y)), 20, (0,255,0), -1)
    #cv2.circle(frame, (int(center_x), int(projection_y)), 10, (0,0,255), -1)    

    return frame

    #return frame, center_x, projection_y, left_x_top, right_x_top, left_x_bottom, right_x_bottom


# In[ ]:


cap = cv2.VideoCapture("fire_right_top.mp4")
homo_img = cv2.imread("homo_img.png")

if cap.isOpened():
    while True:
        ret, img=cap.read()
        
        
        img=cv2.flip(img, 0)
        
        #frame, center_x, center_y, height, width = yolo(frame=img, size=size_list[1], score_threshold=0.4, nms_threshold=0.4)
        
        center_x=700
        center_y=60
        
        height=480
        width=854
        
        cv2.line(img, (0, 100), (width, 100), (0,0,255), 1)
        
        frame2 = line_detection(img,center_x, center_y, height, width)

        #frame3, pos_x, pos_y = f_homography(frame2, homo_img, left_x_top, right_x_top, left_x_bottom, right_x_bottom, height, center_x, projection_y)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    
        if ret:
            #cv2.imshow("Output_Yolo", frame)
            #cv2_imshow(frame)
            #print(center_x, center_y)
            cv2.imshow("Output_line", frame2)
            #cv2_imshow(frame2)

            #cv2.imshow("Output_Homo", frame3)
            #cv2_imshow(frame3)
            #print(pos_x, pos_y)
            #send_arduino(pos_x, pos_y)

            cv2.waitKey(0)

else:
    print("can't open video.")

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import math
import serial



arduino = serial.Serial('COM6', 9600)

area_x=590
area_y=1000

pos_x=100
pos_y=200

sprinkler_2_x=area_x*0.5
sprinkler_2_y=area_y*0.75

angle_2=math.atan2(pos_x-sprinkler_2_x, sprinkler_2_y-pos_y)*180/math.pi

length_2=math.sqrt(pow(pos_x-sprinkler_2_x,2)+pow(sprinkler_2_y-pos_y,2))

 
    
if (angle_2 < 0):
    angle_2 += 360
        

print(angle_2)

b=int(angle_2)
a="Q"+ str(b)
c= a.encode('utf-8')

while(1):
    arduino.write(c)
    print(a)


# In[ ]:


import cv2
import math
import serial

arduino = serial.Serial('COM6', 9600)

area_x=590
area_y=1000

pos_x=100
pos_y=200

sprinkler_2_x=area_x*0.5
sprinkler_2_y=area_y*0.75

angle_2=math.atan2(pos_x-sprinkler_2_x, sprinkler_2_y-pos_y)*180/math.pi

    
if (angle_2 < 0):
    angle_2 += 360

    
print(angle_2)

b=int(angle_2)
a="Q"+ str(b)
c= a.encode('utf-8')

cap = cv2.VideoCapture(1)
homo_img = cv2.imread("homo_img.png")

if cap.isOpened():
    while True:
        ret, img=cap.read()
        
        img=cv2.flip(img, 0)
                
        center_x=700
        center_y=60
        
        height=480
        width=854
        
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    
        if ret:
            cv2.imshow("Output_line", img)
            arduino.write(c)
            print(a)
            cv2.waitKey(0)

else:
    print("can't open video.")

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




