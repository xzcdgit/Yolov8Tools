import os 
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.ticker import MultipleLocator 
from datetime import datetime
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r'C:\Windows\Fonts\长仿宋体.ttf')


if __name__ == '__main__':
    files_folder_path = r'C:\Users\01477483\Desktop\临时文件\数据统计\imgs'
    out_folder_path = r'C:\Users\01477483\Desktop\临时文件\数据统计\output'
    files = os.listdir(files_folder_path)
    cord_list = []
    new_list = []
    for file_full_name in files:
        suffix = file_full_name.split('.')[-1]
        file_name = file_full_name[:-(len(suffix)+1)]
        stses = file_name.split('_')
        if len(stses) > 3:
            continue
        prefix = stses[0]
        if stses[1] == 'True':
            person_flag = 1
        else:
            person_flag = 0
        time_stamp = int(stses[2])
        if time_stamp < 1720911600000 or time_stamp > 1720998000000:
            continue
        cord_list.append(file_full_name)
        new_list.append([time_stamp, person_flag])
    for file_name in cord_list:
        file_full_path = files_folder_path + '\\' + file_name
        shutil.copy2(file_full_path, out_folder_path)
    new_list.sort(key=lambda x:x[0])
    print(len(new_list))

    new_list = np.array(new_list)
    non_compliance = np.sum(new_list[:,1])
    compliance = len(new_list)-non_compliance
    print('不合规数：', non_compliance)
    print('合规数', compliance)

    day_count = 0
    night_count = 0

    day_comliance = 0
    day_non_comliance = 0
    night_comliance = 0
    night_non_comliance = 0
    for data in new_list:
        if data[0] < 1720954800000:
            day_non_comliance += data[1]
            day_count += 1
        else:
            night_non_comliance += data[1]
            night_count += 1
    day_comliance = day_count - day_non_comliance
    night_comliance = night_count - night_non_comliance
    print(day_count, day_comliance, day_non_comliance)
    print(night_count, night_comliance, night_non_comliance)

    


    '''
    xs = new_list[:, 0]
    xs = np.round(xs - 1720886400000)/1000/60/60
    ys = np.diff(xs)
    ys = np.insert(ys, 0, 0) * 60
    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.figure(figsize=(128,24), dpi=80)
    plt.title('Line Chart(green:compliance red:non-compliance)')
    plt.xlabel('time(hour)')
    plt.ylabel('time interval(minute)')
    plt.plot(xs, ys)
    for index,x in enumerate(xs):
        if new_list[index,1] == 1:
            color = 'red'
        else:
            color = 'green'
        plt.scatter(x, ys[index], c=color)
    plt.savefig('test.png')
    plt.show()
    print("finished")
    '''
