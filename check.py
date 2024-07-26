from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import time
import os

import cv2

#按yolo的规则缩放图片
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img
    

#速度测试
def val_check(model_path:str, check_path:str, test_num:int):
    # 类型判定
    if os.path.isdir(check_path):
        model = YOLO(model_path)        
        files = os.listdir(check_path)
        imgs = []
        for index in range(test_num):
            file = files[index]
            if file.split(".")[-1] != 'jpg':
                continue
            img = cv2.imread(check_path+"\\"+file)
            imgs.append(img)
        print('imgs num:{}'.format(len(imgs)))
        while True:
            st_time = time.time()
            results = model.predict(imgs, device = '0')
            print(time.time()-st_time)



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
    model_path = r"D:\Code\Python\DeepLearning\yolov8\runs\detect\train25\weights\best.pt"
    imgs_path = r"D:\Code\Python\DeepLearning\yolov8\dataset\workers\images\train"
    save_path = r"C:\Users\24225\Desktop\临时"
    #check(model_path, imgs_path, save_path)
    val_check(model_path, imgs_path, 10)
    