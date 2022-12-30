import cv2
import numpy as np
import time
import cvzone
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector # CVZONE 包好的 HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector

model  = load_model('./model/keras_model.h5')
labels = open('./model/labels.txt', 'r').readlines()
hand_detector = HandDetector(detectionCon=0.8, maxHands=1)
face_detector = FaceMeshDetector(maxFaces=1)

WIDTH = 1280
HEIGHT = 720
HAND_BIAS = 200

def convert_to_white_background_img(image, bbox, offset=20, image_size = 300):  
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
    return white_background_img

def count_down(img, count_down_time=5):
    count_down_start = time.time() # 倒數開始
    # 等拍照的迴圈
    while True:
        delta_time = time.time() - count_down_start
        cvzone.putTextRect(
            img, f'Count Down: {count_down_time - int(delta_time)} s', 
            pos=(300, 300),
            scale=2,
            offset=20
        )
        cv2.imshow('Webcam Image', img)
        cv2.waitKey(10)
        if count_down_time - delta_time < 0:
            break

def classify_gesture(image):
    probabilities = model.predict(image)
    label = labels[np.argmax(probabilities)]
    most_possible_one_prob = max(probabilities[0])
    most_possible_one_prob = int(most_possible_one_prob * 100)
    most_possible_gesture = label.split()[-1]
    return most_possible_gesture, most_possible_one_prob

def find_face_mesh_boundary(face_info, img_w=WIDTH, img_h=HEIGHT):
    # - 上面：10 # - 下面：152 # - 左邊：234 # - 右邊：454
    # boundary ~= margin
    margin = 100
    _, up_y = face_info[10]
    _, down_y = face_info[152]
    left_x, _ = face_info[234]
    right_x, _ = face_info[454]
    
    face_x_min = max(left_x - margin, 0)
    face_y_min = max(up_y - margin, 0)
    face_x_max = min(right_x + margin, img_w)
    face_y_max = min(down_y + margin, img_h)
    return face_x_min, face_y_min, face_x_max, face_y_max


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    camera.set(3, WIDTH)
    camera.set(4, HEIGHT)
    while True:
        success, image = camera.read()
        if success:
            img_clean = image.copy()
            hands, hands_image = hand_detector.findHands(image)
            if hands:
                hand = hands[0]
                bbox = hand['bbox']

                # 取得手的中心點座標
                center_x, center_y = hand['center'] 
                if HAND_BIAS < center_x < WIDTH - HAND_BIAS and \
                HAND_BIAS < center_y < HEIGHT - HAND_BIAS :
                    white_background_img = convert_to_white_background_img(hands_image, bbox)
                    gesture, prob = classify_gesture(white_background_img)
                    
                    if prob > 80:
                        cvzone.putTextRect(img_clean, f"{gesture}-{prob}%", (x,y-50))
                        if gesture == 'A':
                            count_down(img_clean)  # 開始倒數，把倒數時間放在畫面上。
                            # 辨識臉，重新讀一次camera的圖像
                            success, img_clean = camera.read()
                            if success:
                                face_img, faces = face_detector.findFaceMesh(img_clean, draw=False)
                                if faces:
                                    face = faces[0]
                                    face_x_min, face_y_min, face_x_max, face_y_max = find_face_mesh_boundary(face)
                                    face_img = face_img[
                                        face_y_min:face_y_max, 
                                        face_x_min:face_x_max
                                    ]
                                    cv2.imwrite(f"./photo/{time.time()}.jpg", face_img)
                                    cv2.waitKey(1000)

            cv2.imshow('Webcam Image', img_clean)
            keyboard_input = cv2.waitKey(1)
            if keyboard_input == ord('q'):
                break

    camera.release()
    cv2.destroyAllWindows()