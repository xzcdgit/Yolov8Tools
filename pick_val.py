import os
import shutil
import random
#随机挑选xx%的数据作为测试集

if __name__ == "__main__":
    data_path = r"C:\Code\Python\Yolov8Tools\datasets\men_heng_liang_liu_shui_xian"
    ori_data_path = r"C:\Users\24225\Desktop\2025-01-03\mege"
    train_imgs_path = data_path + r"\images\train"
    train_labels_path = data_path + r"\labels\train"
    val_imgs_path = data_path + r"\images\val"
    val_labels_path = data_path + r"\labels\val"
    files = os.listdir(ori_data_path)
    print(files)
    flies_num = len(files)
    img_names = []
    label_names = []
    for file in files:
        suffix = file.split('.')[-1]
        if suffix == 'jpg':
            img_names.append(file)
        elif suffix == 'txt':
            label_names.append(file)
    
    sample_size = len(img_names) // 10  # 随机取xx%的数据作为测试集
    random_part = random.sample(img_names, sample_size)
    remaining_part = [item for item in img_names if item not in random_part]

    print("随机部分:", random_part)
    print("剩余部分:", remaining_part)
    
    for file in random_part:
        img_full_path = ori_data_path+'\\'+file
        label_full_path = img_full_path[:-3]+'txt'
        shutil.copy(img_full_path,val_imgs_path)
        shutil.copy(label_full_path,val_labels_path)
        
    for file in remaining_part:
        img_full_path = ori_data_path+'\\'+file
        label_full_path = img_full_path[:-3]+'txt'
        shutil.copy(img_full_path,train_imgs_path)
        shutil.copy(label_full_path,train_labels_path)
    
    
    
    