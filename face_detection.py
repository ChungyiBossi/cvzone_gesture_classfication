import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
cap = cv2.VideoCapture(0)  # 0 = 攝像頭編號
face_detector = FaceMeshDetector(maxFaces=1)

while True:   # 用無窮迴圈不斷讀取圖像
    success, img = cap.read() # 讀取圖像：回傳（成功與否, 讀到的圖）
    img, faces = face_detector.findFaceMesh(img, draw=False)
    if success == True:       # 讀取成功才顯示圖片]
        if faces:
            face = faces[0]
            left_eye = face[145]
            right_eye = face[374]
            # cv2.line(
            #     img=img,
            #     pt1=left_eye,
            #     pt2=right_eye,
            #     thickness=3,
            #     color=(0, 200, 0)
            # )
            # cv2.circle(img=img, center=left_eye, radius=10, color=(200, 0, 0))
            # cv2.circle(img=img, center=right_eye, radius=10, color=(200, 0, 0))
            
            # ## Find focus
            w, _ = face_detector.findDistance(left_eye, right_eye) # pixel
            # d = 50  # cm
            # W = 6.3 # cm
            # ## w : f = W : d, w/f=W/d, f=w*d/W
            # f = w*d/W # pixel
            # print(f)

            f = 1675 # pixel
            W = 6.3 # cm
            d = (f * W)/w # cm
            print(d)

            forhead_x, forhead_y = face[10]
            cvzone.putTextRect(
                img=img,
                text=f"Face Depth:{int(d)}",
                pos=(forhead_x-20, forhead_y-50)
            )


        cv2.imshow('Show camera.', img) # 顯示圖片到『show camera』的新視窗 
    
    key = cv2.waitKey(1) # 1ms