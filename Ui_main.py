# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Code\Python\Yolov8Tools-ui\main.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_7 = QtWidgets.QLabel(self.groupBox_3)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 0, 0, 1, 1)
        self.pushButton_path_labelimg = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_path_labelimg.setObjectName("pushButton_path_labelimg")
        self.gridLayout_4.addWidget(self.pushButton_path_labelimg, 0, 1, 1, 1)
        self.lineEdit_path_soft = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_path_soft.setObjectName("lineEdit_path_soft")
        self.gridLayout_4.addWidget(self.lineEdit_path_soft, 0, 2, 1, 1)
        self.pushButton_run_labelimg = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_run_labelimg.setObjectName("pushButton_run_labelimg")
        self.gridLayout_4.addWidget(self.pushButton_run_labelimg, 0, 3, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_3, 0, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 0, 1, 1)
        self.pushButton_path_choice_img = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_path_choice_img.setObjectName("pushButton_path_choice_img")
        self.gridLayout_3.addWidget(self.pushButton_path_choice_img, 0, 1, 1, 1)
        self.lineEdit_path_choice_img = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_path_choice_img.setObjectName("lineEdit_path_choice_img")
        self.gridLayout_3.addWidget(self.lineEdit_path_choice_img, 0, 2, 1, 2)
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 1, 0, 1, 1)
        self.pushButton_path_choice_label = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_path_choice_label.setObjectName("pushButton_path_choice_label")
        self.gridLayout_3.addWidget(self.pushButton_path_choice_label, 1, 1, 1, 1)
        self.lineEdit_path_choice_label = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_path_choice_label.setObjectName("lineEdit_path_choice_label")
        self.gridLayout_3.addWidget(self.lineEdit_path_choice_label, 1, 2, 1, 2)
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 2, 0, 1, 1)
        self.pushButton_path_dataset = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_path_dataset.setObjectName("pushButton_path_dataset")
        self.gridLayout_3.addWidget(self.pushButton_path_dataset, 2, 1, 1, 1)
        self.lineEdit_path_dataset = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_path_dataset.setObjectName("lineEdit_path_dataset")
        self.gridLayout_3.addWidget(self.lineEdit_path_dataset, 2, 2, 1, 2)
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 3, 0, 1, 1)
        self.spinBox_ratio_test = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox_ratio_test.setPrefix("")
        self.spinBox_ratio_test.setMaximum(30)
        self.spinBox_ratio_test.setProperty("value", 10)
        self.spinBox_ratio_test.setObjectName("spinBox_ratio_test")
        self.gridLayout_3.addWidget(self.spinBox_ratio_test, 3, 1, 1, 1)
        self.checkBox_auto_delete = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_auto_delete.setChecked(True)
        self.checkBox_auto_delete.setObjectName("checkBox_auto_delete")
        self.gridLayout_3.addWidget(self.checkBox_auto_delete, 3, 2, 1, 1)
        self.pushButton_run_pick = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_run_pick.setObjectName("pushButton_run_pick")
        self.gridLayout_3.addWidget(self.pushButton_run_pick, 3, 3, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_2, 2, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_5.addWidget(self.label, 0, 0, 1, 1)
        self.pushButton_path_rename_img = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_path_rename_img.setObjectName("pushButton_path_rename_img")
        self.gridLayout_5.addWidget(self.pushButton_path_rename_img, 0, 1, 1, 1)
        self.lineEdit_path_rename_img = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_path_rename_img.setObjectName("lineEdit_path_rename_img")
        self.gridLayout_5.addWidget(self.lineEdit_path_rename_img, 0, 2, 1, 1)
        self.pushButton_run_rename = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_run_rename.setObjectName("pushButton_run_rename")
        self.gridLayout_5.addWidget(self.pushButton_run_rename, 0, 3, 2, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_5.addWidget(self.label_2, 1, 0, 1, 1)
        self.pushButton_path_rename_label = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_path_rename_label.setObjectName("pushButton_path_rename_label")
        self.gridLayout_5.addWidget(self.pushButton_path_rename_label, 1, 1, 1, 1)
        self.lineEdit_path_rename_label = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_path_rename_label.setObjectName("lineEdit_path_rename_label")
        self.gridLayout_5.addWidget(self.lineEdit_path_rename_label, 1, 2, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_8 = QtWidgets.QLabel(self.tab_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_8.addWidget(self.label_8, 0, 0, 1, 1)
        self.groupBox_6 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.radioButton_train_restart = QtWidgets.QRadioButton(self.groupBox_6)
        self.radioButton_train_restart.setChecked(True)
        self.radioButton_train_restart.setObjectName("radioButton_train_restart")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.radioButton_train_restart)
        self.gridLayout_6.addWidget(self.radioButton_train_restart, 0, 0, 1, 1)
        self.checkBox_train_pretrained = QtWidgets.QCheckBox(self.groupBox_6)
        self.checkBox_train_pretrained.setChecked(True)
        self.checkBox_train_pretrained.setObjectName("checkBox_train_pretrained")
        self.gridLayout_6.addWidget(self.checkBox_train_pretrained, 0, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox_6)
        self.label_9.setObjectName("label_9")
        self.gridLayout_6.addWidget(self.label_9, 0, 2, 1, 1)
        self.spinBox_val_epoch = QtWidgets.QSpinBox(self.groupBox_6)
        self.spinBox_val_epoch.setMaximum(10000)
        self.spinBox_val_epoch.setProperty("value", 300)
        self.spinBox_val_epoch.setObjectName("spinBox_val_epoch")
        self.gridLayout_6.addWidget(self.spinBox_val_epoch, 0, 3, 1, 1)
        self.radioButton_train_resume = QtWidgets.QRadioButton(self.groupBox_6)
        self.radioButton_train_resume.setChecked(False)
        self.radioButton_train_resume.setObjectName("radioButton_train_resume")
        self.buttonGroup.addButton(self.radioButton_train_resume)
        self.gridLayout_6.addWidget(self.radioButton_train_resume, 1, 0, 1, 2)
        self.pushButton_run_train = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_run_train.setObjectName("pushButton_run_train")
        self.gridLayout_6.addWidget(self.pushButton_run_train, 1, 3, 1, 1)
        self.gridLayout_8.addWidget(self.groupBox_6, 2, 0, 1, 3)
        self.label_11 = QtWidgets.QLabel(self.tab_2)
        self.label_11.setObjectName("label_11")
        self.gridLayout_8.addWidget(self.label_11, 1, 0, 1, 1)
        self.lineEdit_path_yaml = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_path_yaml.setObjectName("lineEdit_path_yaml")
        self.gridLayout_8.addWidget(self.lineEdit_path_yaml, 1, 2, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.progressBar_train = QtWidgets.QProgressBar(self.groupBox_4)
        self.progressBar_train.setProperty("value", 0)
        self.progressBar_train.setObjectName("progressBar_train")
        self.gridLayout_7.addWidget(self.progressBar_train, 0, 0, 1, 4)
        self.textBrowser_train = QtWidgets.QTextBrowser(self.groupBox_4)
        self.textBrowser_train.setObjectName("textBrowser_train")
        self.gridLayout_7.addWidget(self.textBrowser_train, 1, 0, 1, 4)
        self.label_10 = QtWidgets.QLabel(self.groupBox_4)
        self.label_10.setObjectName("label_10")
        self.gridLayout_7.addWidget(self.label_10, 2, 0, 1, 1)
        self.label_train_time = QtWidgets.QLabel(self.groupBox_4)
        self.label_train_time.setObjectName("label_train_time")
        self.gridLayout_7.addWidget(self.label_train_time, 2, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.groupBox_4)
        self.label_12.setObjectName("label_12")
        self.gridLayout_7.addWidget(self.label_12, 2, 2, 1, 1)
        self.label_trian_resum_time = QtWidgets.QLabel(self.groupBox_4)
        self.label_trian_resum_time.setObjectName("label_trian_resum_time")
        self.gridLayout_7.addWidget(self.label_trian_resum_time, 2, 3, 1, 1)
        self.gridLayout_8.addWidget(self.groupBox_4, 3, 0, 1, 3)
        self.lineEdit_path_model = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_path_model.setObjectName("lineEdit_path_model")
        self.gridLayout_8.addWidget(self.lineEdit_path_model, 0, 2, 1, 1)
        self.pushButton_choice_model = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_choice_model.setObjectName("pushButton_choice_model")
        self.gridLayout_8.addWidget(self.pushButton_choice_model, 0, 1, 1, 1)
        self.pushButton_choice_yaml = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_choice_yaml.setObjectName("pushButton_choice_yaml")
        self.gridLayout_8.addWidget(self.pushButton_choice_yaml, 1, 1, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label_16 = QtWidgets.QLabel(self.tab_3)
        self.label_16.setObjectName("label_16")
        self.gridLayout_9.addWidget(self.label_16, 0, 0, 1, 1)
        self.pushButton_path_check_model = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_path_check_model.setObjectName("pushButton_path_check_model")
        self.gridLayout_9.addWidget(self.pushButton_path_check_model, 0, 1, 1, 1)
        self.lineEdit_path_check_model = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_path_check_model.setObjectName("lineEdit_path_check_model")
        self.gridLayout_9.addWidget(self.lineEdit_path_check_model, 0, 2, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.tab_3)
        self.label_14.setObjectName("label_14")
        self.gridLayout_9.addWidget(self.label_14, 1, 0, 1, 1)
        self.pushButton_path_check = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_path_check.setObjectName("pushButton_path_check")
        self.gridLayout_9.addWidget(self.pushButton_path_check, 1, 1, 1, 1)
        self.lineEdit_path_check = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_path_check.setObjectName("lineEdit_path_check")
        self.gridLayout_9.addWidget(self.lineEdit_path_check, 1, 2, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.tab_3)
        self.label_15.setObjectName("label_15")
        self.gridLayout_9.addWidget(self.label_15, 2, 0, 1, 1)
        self.pushButton_path_save = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_path_save.setObjectName("pushButton_path_save")
        self.gridLayout_9.addWidget(self.pushButton_path_save, 2, 1, 1, 1)
        self.lineEdit_path_save = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_path_save.setObjectName("lineEdit_path_save")
        self.gridLayout_9.addWidget(self.lineEdit_path_save, 2, 2, 1, 1)
        self.pushButton_run_check = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_run_check.setObjectName("pushButton_run_check")
        self.gridLayout_9.addWidget(self.pushButton_run_check, 3, 0, 1, 1)
        self.tabWidget.addTab(self.tab_3, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_3.setTitle(_translate("MainWindow", "标注软件"))
        self.label_7.setText(_translate("MainWindow", "软件路径"))
        self.pushButton_path_labelimg.setText(_translate("MainWindow", "选择路径"))
        self.pushButton_run_labelimg.setText(_translate("MainWindow", "启动"))
        self.groupBox_2.setTitle(_translate("MainWindow", "训练集和测试集随机挑选程序"))
        self.label_3.setText(_translate("MainWindow", "图像文件夹"))
        self.pushButton_path_choice_img.setText(_translate("MainWindow", "选择路径"))
        self.label_4.setText(_translate("MainWindow", "标签文件夹"))
        self.pushButton_path_choice_label.setText(_translate("MainWindow", "选择路径"))
        self.label_5.setText(_translate("MainWindow", "dataset路径"))
        self.pushButton_path_dataset.setText(_translate("MainWindow", "选择路径"))
        self.label_6.setToolTip(_translate("MainWindow", "<html><head/><body><p>选择测试集占测试集+训练集的比例，建议选择10%-15%。</p></body></html>"))
        self.label_6.setWhatsThis(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "测试集比例"))
        self.spinBox_ratio_test.setSuffix(_translate("MainWindow", "%"))
        self.checkBox_auto_delete.setToolTip(_translate("MainWindow", "<html><head/><body><p>勾选该选项，则会先删除dataset文件夹下的所有文件，然后重新创建image、label文件夹及其子文件夹train和val。</p></body></html>"))
        self.checkBox_auto_delete.setText(_translate("MainWindow", "自动删除"))
        self.pushButton_run_pick.setText(_translate("MainWindow", "运行"))
        self.groupBox.setToolTip(_translate("MainWindow", "<html><head/><body><p>将命名不规范的图像和标签文件规范化命名，会自动对应图像和标签文件，只有图像或者标签文件也能运行。</p></body></html>"))
        self.groupBox.setTitle(_translate("MainWindow", "重命名工具"))
        self.label.setText(_translate("MainWindow", "图像文件夹"))
        self.pushButton_path_rename_img.setText(_translate("MainWindow", "选择路径"))
        self.pushButton_run_rename.setText(_translate("MainWindow", "运行"))
        self.label_2.setText(_translate("MainWindow", "标签文件夹"))
        self.pushButton_path_rename_label.setText(_translate("MainWindow", "选择路径"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "原始数据处理"))
        self.label_8.setText(_translate("MainWindow", "模型地址"))
        self.groupBox_6.setTitle(_translate("MainWindow", "训练参数"))
        self.radioButton_train_restart.setText(_translate("MainWindow", "重新训练"))
        self.checkBox_train_pretrained.setText(_translate("MainWindow", "权值预载"))
        self.label_9.setText(_translate("MainWindow", "训练轮数"))
        self.radioButton_train_resume.setText(_translate("MainWindow", "继续训练"))
        self.pushButton_run_train.setText(_translate("MainWindow", "开始训练"))
        self.label_11.setToolTip(_translate("MainWindow", "<html><head/><body><p>选择yaml文件</p></body></html>"))
        self.label_11.setText(_translate("MainWindow", "数据地址"))
        self.groupBox_4.setTitle(_translate("MainWindow", "训练结果"))
        self.label_10.setText(_translate("MainWindow", "已用时间:"))
        self.label_train_time.setText(_translate("MainWindow", "----"))
        self.label_12.setText(_translate("MainWindow", "预估剩余时间:"))
        self.label_trian_resum_time.setText(_translate("MainWindow", "----"))
        self.pushButton_choice_model.setText(_translate("MainWindow", "选择文件"))
        self.pushButton_choice_yaml.setText(_translate("MainWindow", "选择文件"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "训练"))
        self.label_16.setText(_translate("MainWindow", "测试模型路径"))
        self.pushButton_path_check_model.setText(_translate("MainWindow", "选择路径"))
        self.label_14.setText(_translate("MainWindow", "测试素材路径"))
        self.pushButton_path_check.setText(_translate("MainWindow", "选择路径"))
        self.label_15.setText(_translate("MainWindow", "素材保存路径"))
        self.pushButton_path_save.setText(_translate("MainWindow", "选择路径"))
        self.pushButton_run_check.setText(_translate("MainWindow", "运行"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "测试"))
