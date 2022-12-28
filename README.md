12/27 流程

1. 使用python 內建虛擬環境套件『venv』建立一個新的虛擬環境： python3 -m venv [虛擬環境的名字]
    文件連結： https://docs.python.org/zh-tw/3/tutorial/venv.html
    此步驟完成之後，你會得到一個[虛擬環境的名字]的『資料夾』在當前目錄下。

2. 切換到虛擬環境： 
    mac: source [虛擬環境資料夾路徑]/bin/activate
    win10: [虛擬環境資料夾路徑]\Scripts\activate.bat
    輸入指令後，Command Line 新的輸入行會出現(虛擬環境的名字)，代表你已經切換到乾淨新建立的虛擬環境了！

3. pip 安裝 cvzone： pip install cvzone
    此 pip 是跟著虛擬環境一起被創立的新套件管理者。

4. pip 安裝 mediapipe: pip install mediapipe
    如果找不到『mediapipe』，可以先更新pip，更新完再安裝


5. 假設螢幕內的『5 & 17點的平面距離（像素）』 與 『手跟攝像頭的距離』是『線性關係』：
    理論上我們要求：ax^3 + bx^2 + cx = d


12/28

1. 學會如何搜集資料
    - 切出手部被辨識出來的部分
    - 建出一個正方形的白底圖
    - 把手的部分縮放到白底上，尚未填滿的部分就維持白色
2. Teachable Machine 做訓練
3. 使用這個模型：
    - pip install keras
    - pip install --upgrade tensorflow
