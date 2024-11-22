from ultralytics import YOLO
import os
import shutil
import cv2
import numpy as np

#自动打标函数
def label_set(model_path:str, img_folder_path:str, labels_out_folder_path:str, img_show:bool=False):
    # Load a model
    # Run batched inference on a list of images
    if not os.path.isdir(labels_out_folder_path):
        print("输出文件目录路径错误")
        return
    if os.path.isfile(model_path) and model_path.split(".")[-1]!="pt":
        print("模型路径错误")
        return
    model = YOLO(model_path)  # pretrained YOLOv8n model
    out_str = ""
    class_str = ""
    class_max = 0
    if os.path.isdir(img_folder_path):
        files = os.listdir(img_folder_path)
        files_num = len(files)
        index = 0
        while index < files_num:
            out_str = ""
            file = files[index]
            file_type = file.split(".")[-1]
            file_name = file[:-(len(file_type)+1)]
            if file_type != "jpg":
                continue
            results = model(img_folder_path+"\\"+file)  # return a list of Results objects
            # Process results list
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
                #result.show()  # display to screen
                #result.save(filename="result.jpg")  # save to disk
                xywhns = boxes.xywhn.cpu().numpy()
                clses = boxes.cls.cpu().numpy()
                objs_num = len(clses)
                for i in range(objs_num):
                    out_str = out_str + str(int(clses[i])) + " " + str(int(xywhns[i][0]*1000000)/1000000) + " "+ str(int(xywhns[i][1]*1000000)/1000000) + " "+ str(int(xywhns[i][2]*1000000)/1000000) + " "+ str(int(xywhns[i][3]*1000000)/1000000) + "\n"
                    class_max = max(int(clses[i])+1, class_max)
                im_bgr = result.plot()

                #手动选择保存或者自动全部保存
                if img_show:
                    img = np.asanyarray(im_bgr)
                    img_h,img_w = img.shape[:2]
                    img_show = cv2.resize(img,(img_w//2,img_h//2),interpolation=cv2.INTER_CUBIC)
                    cv2.imshow("test", img_show)
                    key = cv2.waitKey(0)
                    if key == ord('d'):
                        index += 1
                        continue
                    elif key == ord("a"):
                        index -= 1
                        continue
                    elif key == ord("s"):
                        with open(labels_out_folder_path+"\\"+file_name+".txt", 'w') as f:
                            f.write(out_str)
                        shutil.copy(img_folder_path+"\\"+file, labels_out_folder_path+"\\"+file)
                        continue
                    elif key == ord("q"):
                        break
                else:
                    with open(labels_out_folder_path+"\\"+file_name+".txt", 'w') as f:
                        f.write(out_str)
                    index += 1
                    
        class_str = ""
        for i in range(class_max):
            class_str = class_str + str(i) + "\n"
            with open(labels_out_folder_path+"\\"+"classes.txt", 'w') as f:
                f.write(class_str)

    elif os.path.isfile(img_folder_path):
        file_type = img_folder_path.split(".")[-1]
        if file_type != "mp4":
            print("目标文件不是 .mp4 文件")
            return
        video_name = img_folder_path.split("\\")[-1]
        cap = cv2.VideoCapture(img_folder_path)
        frames = []
        index = -1
        while cap.isOpened():
                out_str = ""
                if index < 0:
                    ret, frame = cap.read()
                    if len(frames)>10:
                        frames.pop()
                    frames.insert(0,frame)
                    index = 0
                elif index > len(frames)-1:
                    ret = True
                    index = len(frames)-1
                    print("回溯上限")
                if ret:
                    file_name = str(int(cap.get(cv2.CAP_PROP_POS_MSEC)*1000))
                    results = model(frames[index])  # return a list of Results objects
                    # Process results list
                    for result in results:
                        boxes = result.boxes  # Boxes object for bounding box outputs
                        masks = result.masks  # Masks object for segmentation masks outputs
                        keypoints = result.keypoints  # Keypoints object for pose outputs
                        probs = result.probs  # Probs object for classification outputs
                        obb = result.obb  # Oriented boxes object for OBB outputs
                        #result.show()  # display to screen
                        #result.save(filename="result.jpg")  # save to disk
                        xywhns = boxes.xywhn.cpu().numpy()
                        clses = boxes.cls.cpu().numpy()
                        objs_num = len(clses)
                        for i in range(objs_num):
                            out_str = out_str + str(int(clses[i])) + " " + str(int(xywhns[i][0]*1000000)/1000000) + " "+ str(int(xywhns[i][1]*1000000)/1000000) + " "+ str(int(xywhns[i][2]*1000000)/1000000) + " "+ str(int(xywhns[i][3]*1000000)/1000000) + "\n"
                            class_max = max(int(clses[i]+1), class_max)
                        im_bgr = result.plot()
                        break    

                    #手动选择保存或者自动全部保存
                    if img_show:
                        img = np.asanyarray(im_bgr)
                        img_h,img_w = img.shape[:2]
                        img_show = cv2.resize(img,(img_w//2,img_h//2),interpolation=cv2.INTER_CUBIC)
                        cv2.imshow("test", img_show)
                        key = cv2.waitKey(0)
                        if key == ord('d'):
                            index = index - 1
                            continue
                        elif key == ord("a"):
                            index = index + 1
                            continue
                        elif key == ord("s"):
                            with open(labels_out_folder_path+"\\"+video_name + "_" + file_name+".txt", 'w') as f:
                                f.write(out_str)
                            cv2.imwrite(labels_out_folder_path+"\\"+video_name + "_" + file_name+".jpg", frame)
                            continue
                        elif key == ord("q"):
                            break
                    else:
                        with open(labels_out_folder_path+"\\"+video_name + "_" + file_name+".txt", 'w') as f:
                            f.write(out_str)
                        index += 1
        class_str = ""
        for i in range(class_max):
            class_str = class_str + str(i) + "\n"
            with open(labels_out_folder_path+"\\"+"classes.txt", 'w') as f:
                f.write(class_str)


if __name__ == "__main__":
    code_path = os.path.dirname(os.path.abspath(__file__))
    model_path = r"C:\Users\24225\Desktop\train\person\modle\best.pt"
    img_folder_path = r"C:\Users\24225\Desktop\train\person\imgs"
    labels_out_folder_path = r"C:\Users\24225\Desktop\train\person\labels"
    label_set(model_path, img_folder_path, labels_out_folder_path)
