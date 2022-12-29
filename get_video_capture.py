import cv2

cap = cv2.VideoCapture(0)  # 0 = 攝像頭編號

while True:   # 用無窮迴圈不斷讀取圖像
    success, img = cap.read() # 讀取圖像：回傳（成功與否, 讀到的圖）
    if success == True:       # 讀取成功才顯示圖片]
        cv2.imshow('Show camera.', img) # 顯示圖片到『show camera』的新視窗 
    
    key = cv2.waitKey(1)