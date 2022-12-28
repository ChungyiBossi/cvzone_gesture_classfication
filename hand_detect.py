import cv2
import cvzone
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector # CVZONE 包好的 HandDetector

cap = cv2.VideoCapture(0)  # 0 = 攝像頭編號
cap.set(3, 1280) # 設定解析度：橫向
cap.set(4, 720)  # 設定解析度：縱向

# Relationship
# x is the raw distance (in pixel) 
# y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
A, B, C  = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
print("A、B、C：", A, B, C )

# Detector
detector = HandDetector(detectionCon=0.8, maxHands=1) # 利用 class:HandDetector 創造一個新的物件

while True:   # 用無窮迴圈不斷讀取圖像
    success, img = cap.read() # 讀取圖像：回傳（成功與否, 讀到的圖）
    if success == True:       # 讀取成功才顯示圖片
        hands, detected_img = detector.findHands(img) # 丟圖片給detector，讓其辨識出手在哪裡？
        if hands:
            bbox = hands[0]['bbox'] # 偵測到手的框框
            x, y, w, h = bbox # 取得：左上角的Ｘ、左上角的Y、寬度、高度
            
            lmList = hands[0]['lmList'] # 第一隻手關節的三維座標們
            x1, y1, _ = lmList[5]  # 小拇指接手掌的點
            x2, y2, _ = lmList[17] # 食指接手掌的點


            distance_in_pixel = math.sqrt((x1-x2)**2 + (y1-y2)**2) # 算畫面上的距離
            hand_depth = A * (distance_in_pixel ** 2) + B * distance_in_pixel + C
            print(distance_in_pixel, hand_depth)
            cvzone.putTextRect(
                img=detected_img,
                text=f'{hand_depth}',
                pos=(x,y)  # bbox x,y
            )
        
        
        cv2.imshow('Show camera.', detected_img) # 顯示圖片到『show camera』的新視窗 
    
    key = cv2.waitKey(1)
