import cv2
import numpy as np
import cvzone
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector # CVZONE 包好的 HandDetector

camera = cv2.VideoCapture(1)
model  = load_model('./model/keras_model.h5')
labels = open('./model/labels.txt', 'r').readlines()
detector = HandDetector(detectionCon=0.8, maxHands=1)

WIDTH = 1280
HEIGHT = 720
camera.set(3, WIDTH)
camera.set(4, HEIGHT)
offset = 20
image_size = 300

while True:
    ret, image = camera.read()
    hands, image = detector.findHands(image) 
    if hands:
        bbox = hands[0]['bbox']
        x, y, w, h = bbox

        #### 以下為把被辨識出手的部分切出來，放在白底上
        # cutted img include one-hand
        min_y = max(y-offset, 0)   # y-offset 小於零時，取零
        max_y = min(y + h + offset, HEIGHT) # y + h + offest 大於最大高度時，取最大高度
        min_x = max(x-offset,0)
        max_x = min(x + w + offset, WIDTH)
        cutted_img = image[ min_y:max_y , min_x:max_x]
        cutted_img_y, cutted_img_x, _ = cutted_img.shape

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
        
        #### 以上為把被辨識出手的部分切出來，放在白底上

        # 辨識出『機率』與『標籤』
        white_background_img = cv2.resize(white_background_img, (224, 224), interpolation=cv2.INTER_AREA)
        white_background_img = np.asarray(white_background_img, dtype=np.float32).reshape(1, 224, 224, 3)
        white_background_img = (white_background_img / 127.5) - 1
        probabilities = model.predict(white_background_img)
        label = labels[np.argmax(probabilities)]
        most_possible_one_prob = max(probabilities[0])
        most_possible_one_prob = int(most_possible_one_prob * 100)

        if most_possible_one_prob > 80:
            cvzone.putTextRect(image, f"{label.strip()}-{most_possible_one_prob}%", (x,y-50))
    cv2.imshow('Webcam Image', image)

    keyboard_input = cv2.waitKey(1)
    if keyboard_input == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()