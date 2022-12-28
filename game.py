import cv2
import cvzone
import numpy as np
import math
import time
import random
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

# Time Counter
time_start = time.time()

# Button
color = (255, 0, 0)
score = 0   # 得分
total_time = 30 # 總共的遊戲時間
cx, cy = random.randint(100, 1180), random.randint(100, 620) # 按鈕出現的位置

while True:   # 用無窮迴圈不斷讀取圖像
    success, img = cap.read() # 讀取圖像：回傳（成功與否, 讀到的圖）
    if success == True:       # 讀取成功才顯示圖片
        hands, detected_img = detector.findHands(img) # 丟圖片給detector，讓其辨識出手在哪裡？
        if hands:
            bbox = hands[0]['bbox'] # 偵測到手的框框
            x, y, w, h = bbox         # 取得：左上角的Ｘ、左上角的Y、寬度、高度
            lmList = hands[0]['lmList'] # 第一隻手的三維座標們
            x1, y1, z1 = lmList[5]    # 小拇指接手掌的點
            x2, y2, z2 = lmList[17] # 食指接手掌的點


            distance_in_pixel = math.sqrt((x1-x2)**2 + (y1-y2)**2) # 算畫面上的距離
            hand_depth = A * (distance_in_pixel ** 2) + B * distance_in_pixel + C
            # print(distance_in_pixel, hand_depth)


            # Push Button
            if hand_depth < 40:
                if x < cx < x+w and y < cy < y+h:
                    color = (0, 255, 0) # 綠色
                else:
                    color = (255, 0, 0) # 藍色
            else:
                if color == (0, 255, 0) :
                    cx, cy = random.randint(0, 1280), random.randint(0, 720)
                    score += 1
                color = (255, 0, 0)
        
        
        # Draw Button
        cv2.circle(detected_img, (cx, cy), 30, color, cv2.FILLED)
        cv2.circle(detected_img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(detected_img, (cx, cy), 20, (255, 255, 255), 2)
        cv2.circle(detected_img, (cx, cy), 30, (50, 50, 50), 2)
 
        # Game HUD
        remain_time = int(total_time-(time.time()-time_start))
        if remain_time < 0:
            # 時間到
            break
        
        detected_img = cv2.flip(detected_img, 1)
        
        # Time counter
        cvzone.putTextRect(
            detected_img, f'Time: {remain_time}',
            pos=(1000, 75),
            scale=3,
            offset=20
        )

        # Score
        cvzone.putTextRect(detected_img, f'Score: {str(score).zfill(2)}', 
            pos=(60, 75),
            scale=3,
            offset=20
        )

        # 手深度
        if hands:
            cvzone.putTextRect(
                img=detected_img,
                text=f'{int(hand_depth)}',
                pos=(1280-x-w,y)
            )

        cv2.imshow('Show camera.', detected_img) # 顯示圖片到『show camera』的新視窗 

    key = cv2.waitKey(1)


detected_img = cv2.flip(detected_img, 1)
cvzone.putTextRect(
    detected_img, f'Score: {str(score).zfill(2)}', 
    pos=(500, 300),
    scale=6,
    offset=20
)
cv2.imshow('Show camera.', detected_img) # 顯示圖片到『show camera』的新視窗 
cv2.waitKey(5000)
# cv2.destroyAllWindows()