import os
import cv2
import time
# 将已经打好标签的图像进行切割并移动label框

def img_cut(imgs_path:str, labels_path:str,out_path:str, width_left_ratio:float=0, width_right_ratio:float=1, height_up_ratio:float=0, height_down_ratio:float=1):
    if not os.path.isdir(imgs_path):
        print("图像文件路径错误")
        return
    if not os.path.isdir(labels_path):
        print("标签文件路径错误")
        return
    if not os.path.isdir(out_path):
        print("输出文件路径错误")
        return
    imgs_name = os.listdir(imgs_path)
    #labels = os.listdir(labels_path)
    
    for img_name in imgs_name:
        img_name_prefix = img_name[:-4]
        label_name = img_name_prefix + ".txt"
        label_path = labels_path + "\\" + label_name
        if not os.path.isfile(label_path):
            print("图像 {} 不存在对应的label文件".format(img_name))
            continue
        img_path = imgs_path + "\\" +img_name
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        img_new = img[int(img_height*height_up_ratio):int(img_height*height_down_ratio),int(img_width*width_left_ratio):int(img_width*width_right_ratio)]
        img_new_height, img_new_width = img_new.shape[:2]
        print(img_new_height, img_new_width)
        out_label_text = ""
        with open(label_path,'r',encoding='utf-8') as f:
            label_txt = f.read()
            label_rows = label_txt.split("\n")
            for label_row in label_rows:
                if len(label_row) < 5:
                    continue
                cls, x, y, width, height = label_row.split(" ")
                cls, x, y, width, height = int(cls), float(x), float(y), float(width), float(height)
                ori_left = (x-0.5*width)*img_width
                new_left = max(ori_left-width_left_ratio*img_width,0)
                ori_right = (1-x-0.5*width)*img_width
                new_right = min(ori_right - (1-width_right_ratio)*img_width, img_new_width)
                
                ori_up = (y-0.5*height)*img_height
                new_up = max(ori_up-height_up_ratio*img_height, 0)
                ori_down = (1-y-0.5*height)*img_height
                new_down = min(ori_down - (1-height_down_ratio)*img_height, img_new_height)
                
                new_x_ratio = (new_left + img_new_width - new_right)*0.5/img_new_width
                new_y_ratio = (new_up + img_new_height - new_down)*0.5/img_new_height
                new_width_ratio = (img_new_width-new_right-new_left)/img_new_width
                new_height_ratio = (img_new_height-new_down-new_up)/img_new_height
                
                if new_width_ratio==0 or new_height_ratio==0:
                    continue
                
                out_label_text = out_label_text+str(cls)+" "+str(int(new_x_ratio*1e6)/1e6)+" "+str(int(new_y_ratio*1e6)/1e6)+" "+str(int(new_width_ratio*1e6)/1e6)+" "+str(int(new_height_ratio*1e6)/1e6)+"\n"
        out_name = str(int(time.time()*1000))
        cv2.imwrite(out_path+"\\"+out_name+".jpg", img_new)
        with open(out_path+"\\"+out_name+".txt",'w',encoding='utf-8') as f:
            f.write(out_label_text)
        #print(out_label_text)
        #cv2.imshow("test",img_new)
        #cv2.waitKey(0)

if __name__ == "__main__":
    imgs_path = r"C:\Users\24225\Desktop\train\person\imgs"
    labels_path = r"C:\Users\24225\Desktop\train\person\labels"
    out_path = r"C:\Users\24225\Desktop\train\person\output"
    img_cut(imgs_path, labels_path, out_path, 0.4464, )
    