from ultralytics import YOLO
import os
import cv2
import numpy as np


#自动打标函数
def label_set(model_path:str, img_folder_path:str, labels_out_folder_path:str, img_show:bool=False):
    # Load a model
    model = YOLO(model_path)  # pretrained YOLOv8n model
    # Run batched inference on a list of images
    if not os.path.isdir(img_folder_path):
        print("图像文件目录路径错误")
        return
    if not os.path.isdir(labels_out_folder_path):
        print("label文件输出路径错误")
        return
    class_max = 0
    files = os.listdir(img_folder_path)
    files_num = len(files)
    index = 0
    while index < files_num:
        out_str = ""
        file = files[index]
        file_name = file.split(".")[0]
        file_type = file.split(".")[-1]
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
            break
        #保存classes.txt文件
        with open(labels_out_folder_path+"\\"+"classes.txt", 'w') as f:
            classes_str = ""
            for i in range(class_max):
                classes_str = classes_str + "type"+str(i) + "\n"
            f.write(classes_str)

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
                index += 1
                with open(labels_out_folder_path+"\\"+file_name+".txt", 'w') as f:
                    f.write(out_str)
                continue
            elif key == ord("q"):
                break
        else:
            with open(labels_out_folder_path+"\\"+file_name+".txt", 'w') as f:
                f.write(out_str)
            index += 1

if __name__ == "__main__":
    code_path = os.path.dirname(os.path.abspath(__file__))
    model_path = code_path + "\\model\\board\\best_n.pt"
    img_folder_path = code_path + "\\代码测试图像\\images"
    labels_out_folder_path = code_path + "\\代码测试图像\\labels"
    label_set(model_path, img_folder_path, labels_out_folder_path)
