import cv2
import mediapipe as mp
import time
# from cvzone.PoseModule import PoseDetector

pose = mp.solutions.pose
Poser = pose.Pose()
Drawer = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp.solutions.holistic.POSE_CONNECTIONS
# Poser = PoseDetector()

WIDTH = 1280
HEIGHT = 720
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

frame_start_time = time.time()
while True:
    success, image = cap.read()
    if success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        delta_time = time.time() - frame_start_time
        frame_start_time = time.time()
        fps = 1/(delta_time)

        result = Poser.process(image)
        # lmList, bboxInfo = Poser.findPosition(img, bboxWithHands=False)
        if result.pose_landmarks:
            Drawer.draw_landmarks(image, result.pose_landmarks, POSE_CONNECTIONS)
            for id, lm in enumerate(result.pose_landmarks.landmark):
                pos_x = int(lm.x * WIDTH)
                pos_y = int(lm.y * HEIGHT)
                print(f"{id} - X:{pos_x}, Y:{pos_y}")
                cv2.circle(
                    img=image, 
                    center=(pos_x, pos_y),
                    radius=10,
                    color=(255, 255, 255),
                    thickness=cv2.FILLED
                )
                

        cv2.putText(
            img=image,
            text=f'FPS: {int(fps)}',
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=3,
            color=(255, 0, 0)
        )
    cv2.imshow("Pose Estimation", image)
    key = cv2.waitKey(1)