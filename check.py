from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import time
import os

#检测
def check(model_path:str, check_path:str, save_path:str=""):
    
    # 类型判定
    if os.path.isdir(check_path):
        model = YOLO(model_path)        
        files = os.listdir(check_path)
        index = 0
        while index<len(files):
            file = files[index]
            if file.split(".")[-1] != 'jpg':
                continue
            img = cv2.imread(check_path+"\\"+file)
            results = model(img, device = 0)
            # Visualize the results
            for i, r in enumerate(results):
                # Plot results image
                im_bgr = r.plot()  # BGR-order numpy array
            frame = np.asarray(im_bgr)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
            cv2.imshow('frame', frame)  # 显示读取到的这一帧画面
            key = cv2.waitKey(0)       # 等待一段时间，并且检测键盘输入
            if key == ord('d'):     
                index += 1
            elif key == ord('a'):     
                index -= 1
            elif key == ord('s'):
                if os.path.isdir(save_path):
                    cv2.imwrite(save_path+"\\"+file, img)
            elif key == ord('q'):
                break
        cv2.destroyAllWindows()           # 关闭所有窗口
        
    elif os.path.isfile(check_path):
        if check_path.split(".")[-1] == "mp4":
            model = YOLO(model_path)
            cap = cv2.VideoCapture(check_path)     # 读取视频
            while cap.isOpened():               # 当视频被打开时：
                ret, frame = cap.read()         # 读取视频，读取到的某一帧存储到frame，若是读取成功，ret为True，反之为False
                if ret:                         # 若是读取成功
                    results = model(frame, device = 0)
                    print(results)
                    # Visualize the results
                    for i, r in enumerate(results):
                        # Plot results image
                        im_bgr = r.plot()  # BGR-order numpy array
                    frame = np.asarray(im_bgr)
                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
                    cv2.imshow('frame', frame)  # 显示读取到的这一帧画面
                    key = cv2.waitKey(0)       # 等待一段时间，并且检测键盘输入
                    if key == ord('d'):         # 若是键盘输入'q',则退出，释放视频
                        index += 1
                    elif key == ord('a'):         # 若是键盘输入'q',则退出，释放视频
                        index -= 1
                    elif key == ord('s'):
                        if os.path.isdir(save_path):
                            cv2.imwrite(save_path+"\\"+file, img)
                    elif key == ord('q'):
                        break
                else:
                    cap.release()
            cv2.destroyAllWindows()             # 关闭所有窗口

if __name__ == "__main__":
    #videotest(r"D:\Code\Python\DeepLearning\yolov8\runs\detect\train11\weights\best.pt", r"C:\Users\ADMIN\Desktop\素材\叠板检测\0612\20240611.mp4")
    model_path = r"C:\Code\Python\Yolov8Tools\runs\detect\train4\weights\best.pt"
    imgs_path = r"C:\Code\Python\Yolov8Tools\datasets\workers\images\train"
    save_path = r"C:\Users\24225\Desktop\临时"
    check(model_path, imgs_path, save_path)