import os
import sys
import time
import subprocess
import shutil
import random
import cv2
from ultralytics import YOLO
import numpy as np
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QPixmap, QImage
from Ui_main import Ui_MainWindow


def run_software(path):
    """
    运行软件
    """
    try:
        if os.path.isfile(path):
            subprocess.run(path, check=True)
        elif os.path.isdir(path):
            subprocess.run(f"{path}\labelImg.exe", check=True)
        message = f"软件 {path} 运行成功"
    except subprocess.CalledProcessError as e:
        message = f"运行软件 {path} 时出错: {e}"
    except FileNotFoundError:
        message = f"未找到路径: {path}"
    return message


def rename_images_and_txt(folder1, folder2):
    message = ""
    """
    重命名文件夹1中的图像文件和文件夹2中对应的txt文件
    """
    if not os.path.isdir(folder1):
        message = "图像文件夹夹不存在"
        print(1)
    elif not os.path.isdir(folder2):
        message = "标签文件夹夹不存在"
        print(2)
    else:
        groups = []
        # 获取目标文件夹1中的所有图像文件
        image_files = [
            f
            for f in os.listdir(folder1)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
        # 获取文件夹2中所有和文件夹1中图像文件对应的txt文件
        for i, image_file in enumerate(image_files):
            group = []
            image_path = f"{folder1}\\{image_file}"
            group.append(image_path)
            name = os.path.splitext(image_file)[0]
            find_name = f"{folder2}\\{name}.txt"
            if os.path.exists(find_name):
                group.append(find_name)
            else:
                group.append(None)
            groups.append(group)

        # 重命名文件为临时名字，防止要重命名的文件名重复
        for group in groups:
            for ele in group:
                if ele is not None:
                    os.rename(ele, ele + "tmp")
        # 重命名文件为目标名字
        folder = [folder1, folder2]
        for i, group in enumerate(groups):
            for j, ele in enumerate(group):
                if ele is not None:
                    # name = os.path.splitext(ele)[0]
                    ext = os.path.splitext(ele)[1]
                    new_name = f"{i:06d}{ext}"
                    os.rename(ele + "tmp", folder[j] + "\\" + new_name)
                    message += f"重命名文件: {ele} -> {folder[j]} + '\\' + {new_name}\n"
    return message


def split_and_copy_images(folder1, folder2, folder_a, ratio, auto_delete=False):
    """
    随机抽取部分图像文件以及另一个文件夹下对应的txt并分别复制到目标文件夹
    """
    message = ""

    # 检查文件夹是否存在
    if not os.path.exists(folder1):
        raise FileNotFoundError(f"未找到文件夹: {folder1}")
    if not os.path.exists(folder2):
        raise FileNotFoundError(f"未找到文件夹: {folder2}")
    if not os.path.exists(folder_a):
        raise FileNotFoundError(f"未找到文件夹: {folder_a}")

    if auto_delete:
        # 删除目标文件夹下的所有文件
        shutil.rmtree(folder_a)

    # 创建目标文件夹
    val_images_folder = os.path.join(folder_a, "images", "val")
    train_images_folder = os.path.join(folder_a, "images", "train")
    val_labels_folder = os.path.join(folder_a, "labels", "val")
    train_labels_folder = os.path.join(folder_a, "labels", "train")

    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)

    # 获取所有图像文件
    image_files = [
        f
        for f in os.listdir(folder1)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]
    random.shuffle(image_files)  # 随机打乱文件列表

    # 计算分割点
    split_point = int(len(image_files) * ratio)

    # 分割图像文件
    val_files = image_files[:split_point]
    train_files = image_files[split_point:]

    # 复制图像文件和对应的txt文件
    for image_file in val_files:
        shutil.copy(os.path.join(folder1, image_file), val_images_folder)
        txt_file = os.path.splitext(image_file)[0] + ".txt"
        if os.path.exists(os.path.join(folder2, txt_file)):
            shutil.copy(os.path.join(folder2, txt_file), val_labels_folder)

    for image_file in train_files:
        shutil.copy(os.path.join(folder1, image_file), train_images_folder)
        txt_file = os.path.splitext(image_file)[0] + ".txt"
        if os.path.exists(os.path.join(folder2, txt_file)):
            shutil.copy(os.path.join(folder2, txt_file), train_labels_folder)

    message += "文件复制完成"
    return message


class YOLOModel(QObject):
    train_epoch_start_signal = pyqtSignal(list)
    train_end_signal = pyqtSignal(list)
    def __init__(self):
        super().__init__()
        pass

    def train_yolo(self, model_path, data_path, epoch_count, resume):
        """
        训练YOLO模型
        """
        message = ""
        # 定义模型
        model = YOLO(model_path)
        model.add_callback("on_train_epoch_end", self.get_train_info)
        model.add_callback("on_train_end", self.train_end_info)
        # 训练模型
        if resume:
            message += "继续训练模型"
            results = model.train(
                data=data_path, epochs=epoch_count, device=0, resume=True
            )
        else:
            message += "训练新模型"
            results = model.train(
                data=data_path, epochs=epoch_count, device=0, pretrained=True
            )
        return message, results

    def get_train_info(self, trainer):
        """
        获取训练信息
        """
        self.train_epoch_start_signal.emit(
            [
                trainer.epoch,
                trainer.epochs,
                trainer.epoch_time,
                trainer.metrics,
            ]
        )
        
    def train_end_info(self,trainer):
        """
        训练结束
        """
        self.train_end_signal.emit(
            [
                trainer.epoch,
                trainer.epochs,
                trainer.epoch_time,
                trainer.metrics,
                trainer.save_dir
            ]
        )
        pass

    def check_model(self, model_path: str, check_path: str, save_path: str = ""):
        # 类型判定
        message = ""
        if os.path.isdir(check_path):
            index = 0
            model = YOLO(model_path)
            files = os.listdir(check_path)
            while index < len(files):
                file = files[index]
                if file.split(".")[-1] != "jpg":
                    continue
                img = cv2.imread(check_path + "\\" + file)
                results = model(img, device=0)
                # Visualize the results
                for i, r in enumerate(results):
                    # Plot results image
                    im_bgr = r.plot()  # BGR-order numpy array
                frame = np.asarray(im_bgr)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
                cv2.imshow("frame", frame)  # 显示读取到的这一帧画面
                key = cv2.waitKey(0)  # 等待一段时间，并且检测键盘输入
                if key == ord("d"):
                    index += 1
                elif key == ord("a"):
                    index -= 1
                elif key == ord("s"):
                    if os.path.isdir(save_path):
                        cv2.imwrite(save_path + "\\" + file, img)
                        print("图片保存完成 ", save_path + "\\" + file)
                        message += f"图片保存完成 {save_path}\\{file}\n"
                elif key == ord("q"):
                    break
            cv2.destroyAllWindows()  # 关闭所有窗口

        elif os.path.isfile(check_path):
            if check_path.split(".")[-1] == "mp4":
                frame_index = 0
                frame_number = 0
                frames = []
                model = YOLO(model_path)
                cap = cv2.VideoCapture(check_path)  # 读取视频
                while cap.isOpened():
                    if frame_number == frame_index:
                        ret, frame = cap.read()
                        frames.append(frame)
                        if len(frames) >= 200:
                            frames[len(frames) - 200] = None
                        frame_number += 1
                    frame = frames[frame_index]
                    img = frame
                    results = model(frame, device=0)
                    # Visualize the results
                    for r in results:
                        # Plot results image
                        im_bgr = r.plot()  # BGR-order numpy array
                    frame = np.asarray(im_bgr)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
                    cv2.imshow("frame", frame)  # 显示读取到的这一帧画面

                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        cap.release()
                        break
                    elif key == ord("s"):
                        file_full_name = f"{save_path}\\frame_{frame_index}.jpg"
                        res = cv2.imwrite(file_full_name, img)
                        print(res, "图片保存完成 ", file_full_name)
                        message += f"图片保存完成 {file_full_name}\n"
                    elif key == ord("d"):  # Left arrow key
                        frame_index += 1
                    elif (
                        key == ord("a")
                        and frame_index > frame_number - 200
                        and frame_index > 0
                    ):  # Right arrow key
                        frame_index -= 1
                cv2.destroyAllWindows()  # 关闭所有窗口
        else:
            print("文件类型错误")


class MyApp(QMainWindow, Ui_MainWindow):
    my_yolo = YOLOModel()
    train_thread = None
    check_thread = None

    @staticmethod
    def main():
        app = QApplication(sys.argv)
        myapp = MyApp()
        myapp.show()
        sys.exit(app.exec_())

    def __init__(self):

        super().__init__()
        self.setupUi(self)
        self.set_click_trigger()
        self.connect_signals()
        self.default_path()

    def default_path(self):
        """
        设置默认路径
        """
        self.lineEdit_path_soft.setText(r"C:\Program Files\windows_v1.8.1")
        self.lineEdit_path_rename_img.setText(
            r"C:\Users\24225\Desktop\2025-01-03\截图\通道1"
        )
        self.lineEdit_path_rename_label.setText(
            r"C:\Users\24225\Desktop\2025-01-03\截图\通道1"
        )
        self.lineEdit_path_choice_img.setText(
            r"C:\Users\24225\Desktop\2025-01-03\picture\ch1"
        )
        self.lineEdit_path_choice_label.setText(
            r"C:\Users\24225\Desktop\2025-01-03\label\ch1"
        )
        self.lineEdit_path_dataset.setText(r"C:\Users\24225\Desktop\2025-01-03\dataset")
        self.lineEdit_path_model.setText(r"C:\Code\Python\Yolov8Tools\yolov8s.pt")
        self.lineEdit_path_yaml.setText(
            r"C:\Users\24225\AppData\Local\Programs\Python\Python311\Lib\site-packages\ultralytics\cfg\datasets\men_heng_liang_liu_shui_xian.yaml"
        )
        self.lineEdit_path_check_model.setText(
            r"C:\Code\Python\Yolov8Tools\runs\detect\train12\weights\best.pt"
        )
        self.lineEdit_path_check.setText(
            r"C:\Users\24225\Desktop\2025-01-03\10.70.79.200_01_20250103101736342.mp4"
        )
        self.lineEdit_path_save.setText(r"C:\Users\24225\Desktop\2025-01-03\save")

    def connect_signals(self):
        """
        连接信号
        """
        self.my_yolo.train_epoch_start_signal.connect(self.callback_train_info)
        self.my_yolo.train_end_signal.connect(self.callbcak_train_end)

    def set_click_trigger(self):
        """
        按钮触发事件
        """
        self.choice_path_buttons = [
            self.pushButton_path_labelimg,
            self.pushButton_path_rename_img,
            self.pushButton_path_rename_label,
            self.pushButton_path_choice_img,
            self.pushButton_path_choice_label,
            self.pushButton_path_dataset,
            self.pushButton_path_check_model,
            self.pushButton_path_check,
            self.pushButton_path_save,
        ]
        self.choice_file_buttons = [
            self.pushButton_choice_model,
            self.pushButton_choice_yaml,
        ]
        self.pushButton_run_labelimg.clicked.connect(self.start_labelimg)
        self.pushButton_run_rename.clicked.connect(self.start_rename)
        self.pushButton_run_pick.clicked.connect(self.start_split)
        self.pushButton_run_train.clicked.connect(self.start_train)
        self.pushButton_run_check.clicked.connect(self.start_check)
        for index, button in enumerate(self.choice_path_buttons):
            button.clicked.connect(
                lambda checked, idx=index: self.set_choice_trigger(idx)
            )
        for index, button in enumerate(self.choice_file_buttons):
            button.clicked.connect(
                lambda checked, idx=index: self.set_choice_file_trigger(idx)
            )

    def callback_train_info(self, info):
        """
        显示消息
        """
        epoch, epochs, epoch_time, metrics = info
        epoch += 1
        self.progressBar_train.setValue(int(epoch / epochs * 100))
        
        self.textBrowser_train.append(f"Epoch: {epoch}/{epochs}, EpochTime: {epoch_time}")
        elapsed_time = time.time() - self.train_start_time
        formatted_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        self.label_train_time.setText(f"{formatted_elapsed_time}")
        if epoch <= 1:
            self.label_train_resum_time.setText("计算中...")
        else:
            remaining_time = (epochs - epoch)*epoch_time
            formatted_remaining_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            self.label_train_resum_time.setText(f"{formatted_remaining_time}")
    
    def callbcak_train_end(self, info):
        """
        训练结束
        """
        epoch, epochs, epoch_time, metrics, save_dir = info
        epoch += 1
        if epoch < epochs:
            self.textBrowser_train.append("训练提前完成")
        else:
            self.textBrowser_train.append("训练完成")
        self.textBrowser_train.append(f"模型保存在: {save_dir}")
        self.progressBar_train.setValue(100)
        self.label_train_resum_time.setText("00:00:00")
    
    def set_choice_trigger(self, index):
        """
        选择路径按钮触发事件
        """
        self.lineEdit_paths = [
            self.lineEdit_path_soft,
            self.lineEdit_path_rename_img,
            self.lineEdit_path_rename_label,
            self.lineEdit_path_choice_img,
            self.lineEdit_path_choice_label,
            self.lineEdit_path_dataset,
            self.lineEdit_path_check_model,
            self.lineEdit_path_check,
            self.lineEdit_path_save,
        ]
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            self.lineEdit_paths[index].setText(path)

    def set_choice_file_trigger(self, index):
        """
        选择文件按钮触发事件
        """
        self.lineEdit_files = [self.lineEdit_path_model, self.lineEdit_path_yaml]
        path, _ = QFileDialog.getOpenFileName(self, "选择文件")
        if path:
            self.lineEdit_files[index].setText(path)

    def start_labelimg(self):
        """
        启动labelImg软件
        """
        path = self.lineEdit_path_soft.text()
        message = run_software(path)
        self.statusbar.showMessage(message)

    def start_rename(self):
        """
        重命名文件
        """
        folder1 = self.lineEdit_path_rename_img.text()
        folder2 = self.lineEdit_path_rename_label.text()
        message = rename_images_and_txt(folder1, folder2)
        self.statusbar.showMessage(message)

    def start_split(self):
        """
        分割文件
        """
        folder1 = self.lineEdit_path_choice_img.text()
        folder2 = self.lineEdit_path_choice_label.text()
        folder_a = self.lineEdit_path_dataset.text()
        ratio = float(self.spinBox_ratio_test.value() / 100)
        auto_delete = self.checkBox_auto_delete.isChecked()
        print(folder1, folder2, folder_a, ratio)
        message = split_and_copy_images(folder1, folder2, folder_a, ratio, auto_delete)
        self.statusbar.showMessage(message)

    def start_train(self):
        """
        训练模型
        """
        if self.train_thread is not None and self.train_thread.is_alive():
            self.statusbar.showMessage("训练线程已经运行")
            QMessageBox.information(self, "提示", "训练已经开始，请勿重复运行") #最后的Yes表示弹框的按钮显示为Yes，默认按钮显示为OK,不填QMessageBox.Yes即为默认
        else:
            self.train_start_time = time.time()
            model_path = self.lineEdit_path_model.text()
            data_path = self.lineEdit_path_yaml.text()
            epoch_count = self.spinBox_val_epoch.value()
            resume = self.radioButton_train_resume.isChecked()
            self.train_thread = threading.Thread(
                target=self.my_yolo.train_yolo,
                args=(model_path, data_path, epoch_count, resume),
                daemon=True,
            )
            self.train_thread.start()
            self.statusbar.showMessage("训练线程已经启动")
            self.textBrowser_train.append("开始训练，初始化中...")


    def start_check(self):
        if self.check_thread is not None and self.check_thread.is_alive():
            self.statusbar.showMessage("检测线程正在运行")
        else:
            model_path = self.lineEdit_path_check_model.text()
            check_path = self.lineEdit_path_check.text()
            save_path = self.lineEdit_path_save.text()
            self.check_thread = threading.Thread(
                target=self.my_yolo.check_model,
                args=(model_path, check_path, save_path),
                daemon=True,
            )
            self.check_thread.start()
            self.statusbar.showMessage("检测线程已经启动")


if __name__ == "__main__":
    MyApp.main()
