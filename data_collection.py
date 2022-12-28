import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

cap = cv2.VideoCapture(0)
WIDTH = 1280
HEIGHT = 720
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

detector = HandDetector(maxHands=1)
offset = 20
image_size = 300

img_folder_prefix = './Data'
while True:
    success, img = cap.read()
    hands, detected_img = detector.findHands(img)
    if hands:
        bbox = hands[0]['bbox']
        x, y, w, h = bbox

        # cutted img include one-hand
        min_y = max(y-offset, 0)   # y-offset 小於零時，取零
        max_y = min(y + h + offset, HEIGHT) # y + h + offest 大於最大高度時，取最大高度
        min_x = max(x-offset,0)
        max_x = min(x + w + offset, WIDTH)
        cutted_img = detected_img[ min_y:max_y , min_x:max_x]
        cutted_img_y, cutted_img_x, _ = cutted_img.shape
        cv2.imshow("[Cutted Image]", cutted_img)

        # white background image
        shape = (image_size, image_size, 3) # 長Ｘ寬ＸRGB數值，一個三維矩陣
        white_background_img = np.ones(shape=shape, dtype=np.uint8) * 255  # X 255代表數值為白色：R=255, B=255, G=255

        hw_ratio = cutted_img_y/cutted_img_x  # bbox 的 h:w
        # 假設是500 x 300 (cutted_img_y x cutted_img_x)
        height_margin = 0
        width_margin = 0
        if hw_ratio > 1: # 垂直長
            cutted_img_x = int(cutted_img_x * (image_size/cutted_img_y)) # 等比例縮放 x
            cutted_img_y = 300
            width_margin = int((image_size - cutted_img_x)/2)
        else: # 水平長
            # 假設是 300 x 500 (cutted_img_y x cutted_img_x)
            cutted_img_y = int(cutted_img_y * (image_size/cutted_img_x)) # 等比例縮放 y
            cutted_img_x = 300
            height_margin = int((image_size - cutted_img_y)/2)
        
        cutted_img = cv2.resize(cutted_img, (cutted_img_x, cutted_img_y))
        new_cutted_img_y, new_cutted_img_x, _ = cutted_img.shape
        white_background_img[
            height_margin: height_margin + new_cutted_img_y,  # Y,垂直方向
            width_margin : width_margin  + new_cutted_img_x   # X,水平方向
        ] = cutted_img
        
        cv2.imshow("[White BackGround Image]", white_background_img)

        key  = cv2.waitKey(300)
        if key == ord('A') or key == ord('a'):
            cv2.imwrite(f"{img_folder_prefix}/A/{time.time()}.jpg", white_background_img)
        elif key == ord('B') or key == ord('b'):
            cv2.imwrite(f"{img_folder_prefix}/B/{time.time()}.jpg", white_background_img)
        elif key == ord('C') or key == ord('c'):
            cv2.imwrite(f"{img_folder_prefix}/C/{time.time()}.jpg", white_background_img)

    cv2.imshow("[Data Collect]", detected_img)
    key = cv2.waitKey(1)