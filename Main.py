from ultralytics import YOLO
import os
import cv2
import numpy as np


class Yolov8Tools:
    
    def __init__(self) -> None:
        self.default_device = 0
        self.train_input_model_path = ""
        self.train_dataset_yaml_path = ""
        self.train_epoch = 300
        self.train_output_model_path = ""
    
    #自动打标函数
    def label_set(self, model_path:str, img_folder_path:str, labels_out_folder_path:str, img_show:bool=False):
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

    #训练
    def train(self):
        # Create a new YOLO model from scratch
        model = YOLO(self.train_input_model_path)

        # Load a pretrained YOLO model (recommended for training)
        #model = YOLO("yolov8s.pt")

        # Train the model using the 'coco8.yaml' dataset for 3 epochs
        results = model.train(data=self.train_dataset_yaml_path, epochs=self.train_epoch, device=self.default_device)
        self.train_output_model_path = results

        # Evaluate the model's performance on the validation set
        #results = model.val()

        # Perform object detection on an image using the model
        #results = model("https://ultralytics.com/images/bus.jpg")
        # Export the model to ONNX format
        success = model.export(format="onnx")
        print(success)
    
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
                        elif key == ord('s'):
                            if os.path.isdir(save_path):
                                cv2.imwrite(save_path+"\\"+file, img)
                        elif key == ord('q'):
                            break
                    else:
                        cap.release()
                cv2.destroyAllWindows()             # 关闭所有窗口
