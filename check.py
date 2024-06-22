from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import time
import os

def check(model_path:str, pics_path:str):
    model = YOLO(model_path)
    files = os.listdir(pics_path)
    index = 0
    while index<len(files):
        file = files[index]
        file_type = file.split(".")[-1]
        file_name = file[:(-len(file_type+1))]
        if file_type != 'jpg':
            continue
        img = cv2.imread(pics_path+"\\"+file)
        results = model(img, device = 0)
        # Visualize the results
        for i, r in enumerate(results):
            # Plot results image
            im_bgr = r.plot()  # BGR-order numpy array
        frame = np.asarray(im_bgr)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', frame)  # 显示读取到的这一帧画面
        key = cv2.waitKey(0)       # 等待一段时间，并且检测键盘输入
        if key == ord('d'):         # 若是键盘输入'q',则退出，释放视频
            index += 1
        elif key == ord('a'):         # 若是键盘输入'q',则退出，释放视频
            index -= 1
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()           # 关闭所有窗口
    
def video_check(model_path:str, video_path:str):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)     # 读取视频
    while cap.isOpened():               # 当视频被打开时：
        ret, frame = cap.read()         # 读取视频，读取到的某一帧存储到frame，若是读取成功，ret为True，反之为False
        if ret:                         # 若是读取成功
            results = model(frame, device = 0)
            # Visualize the results
            st_time = time.time()
            for i, r in enumerate(results):
                # Plot results image
                im_bgr = r.plot()  # BGR-order numpy array
            frame = np.asarray(im_bgr)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('frame', frame)  # 显示读取到的这一帧画面
            key = cv2.waitKey(1)       # 等待一段时间，并且检测键盘输入
            if key == ord('q'):         # 若是键盘输入'q',则退出，释放视频
                cap.release()           # 释放视频
                break
        else:
            cap.release()
    cv2.destroyAllWindows()             # 关闭所有窗口


if __name__ == "__main__":
    #videotest(r"D:\Code\Python\DeepLearning\yolov8\runs\detect\train11\weights\best.pt", r"C:\Users\ADMIN\Desktop\素材\叠板检测\0612\20240611.mp4")
    pic_check(r"D:\Code\Python\DeepLearning\yolov8\runs\detect\train2\weights\best.pt", r"C:\Users\ADMIN\Desktop\素材\叠板检测\0618\test")